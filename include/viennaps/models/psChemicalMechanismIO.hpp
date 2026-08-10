#pragma once

#include "psSurfaceChemistry.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Read a ChemicalMechanism from a mechanism file (the "IR").
//
// The reaction file is the input to a process; this is what lets C++ consume it
// without the Python toolchain. ViennaChem writes the file and this reads it
// back, so a C++ example is the same two lines as a Python one:
//
//     auto mech = readChemicalMechanism<double>("diamond.mechanism.json");
//     SurfaceChemistry<double, 2> model(mech);
//
// The reader below is deliberately small and self-contained. The format is our
// own, flat, and always machine-written, so a dependency (and a network fetch
// at configure time) would buy little. It handles the whole of JSON except
// what a mechanism file never contains.
//
// This is the ONLY reader of the format: Python reaches it through
// `ps.ChemicalMechanism.fromJSON`, so there is one implementation rather than
// one per language. `ViennaChem/tests/test_loader.py` checks it against the
// reference evaluator on every reaction file, so a misread fails a test rather
// than showing up as a wrong simulation.

namespace viennaps {

// The mechanism-data format this reader understands. ViennaChem stamps the
// version it wrote into every file; a version this reader does not know must be
// refused rather than guessed at, because a silently misread mechanism is a
// simulation of the wrong chemistry.
inline constexpr int chemicalMechanismSchemaVersion = 1;

namespace impl {

// --- a minimal JSON value ------------------------------------------------

class JsonValue;
using JsonPtr = std::shared_ptr<JsonValue>;

class JsonValue {
public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.;
  std::string text;
  std::vector<JsonPtr> array;
  std::map<std::string, JsonPtr> object;

  bool isNull() const { return type == Type::Null; }

  // Member access. `at` requires the key; `get` returns null when absent, which
  // is what an optional part of the schema needs.
  const JsonPtr get(const std::string &key) const {
    auto it = object.find(key);
    return it == object.end() ? nullptr : it->second;
  }

  const JsonValue &at(const std::string &key) const {
    auto it = object.find(key);
    if (it == object.end())
      throw std::runtime_error("mechanism file: missing key '" + key + "'");
    return *it->second;
  }

  double num(const std::string &key, double fallback) const {
    auto v = get(key);
    return (!v || v->isNull()) ? fallback : v->number;
  }

  bool flag(const std::string &key, bool fallback = false) const {
    auto v = get(key);
    return (!v || v->isNull()) ? fallback : v->boolean;
  }

  std::string str(const std::string &key, const std::string &fallback = "") const {
    auto v = get(key);
    return (!v || v->isNull()) ? fallback : v->text;
  }
};

class JsonParser {
  const std::string &s;
  size_t i = 0;

public:
  explicit JsonParser(const std::string &text) : s(text) {}

  JsonPtr parse() {
    auto v = value();
    skipSpace();
    if (i != s.size())
      fail("trailing characters");
    return v;
  }

private:
  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error("mechanism file: " + what + " at offset " +
                             std::to_string(i));
  }

  void skipSpace() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
  }

  bool literal(const char *word) {
    const size_t n = std::char_traits<char>::length(word);
    if (s.compare(i, n, word) != 0)
      return false;
    i += n;
    return true;
  }

  JsonPtr value() {
    skipSpace();
    if (i >= s.size())
      fail("unexpected end of file");
    switch (s[i]) {
    case '{':
      return objectValue();
    case '[':
      return arrayValue();
    case '"':
      return stringValue();
    default:
      break;
    }
    auto v = std::make_shared<JsonValue>();
    if (literal("true")) {
      v->type = JsonValue::Type::Bool;
      v->boolean = true;
      return v;
    }
    if (literal("false")) {
      v->type = JsonValue::Type::Bool;
      v->boolean = false;
      return v;
    }
    if (literal("null"))
      return v; // Null
    return numberValue();
  }

  JsonPtr objectValue() {
    auto v = std::make_shared<JsonValue>();
    v->type = JsonValue::Type::Object;
    ++i; // '{'
    skipSpace();
    if (i < s.size() && s[i] == '}') {
      ++i;
      return v;
    }
    while (true) {
      skipSpace();
      if (i >= s.size() || s[i] != '"')
        fail("expected a key");
      const std::string key = stringValue()->text;
      skipSpace();
      if (i >= s.size() || s[i] != ':')
        fail("expected ':'");
      ++i;
      v->object[key] = value();
      skipSpace();
      if (i < s.size() && s[i] == ',') {
        ++i;
        continue;
      }
      if (i < s.size() && s[i] == '}') {
        ++i;
        return v;
      }
      fail("expected ',' or '}'");
    }
  }

  JsonPtr arrayValue() {
    auto v = std::make_shared<JsonValue>();
    v->type = JsonValue::Type::Array;
    ++i; // '['
    skipSpace();
    if (i < s.size() && s[i] == ']') {
      ++i;
      return v;
    }
    while (true) {
      v->array.push_back(value());
      skipSpace();
      if (i < s.size() && s[i] == ',') {
        ++i;
        continue;
      }
      if (i < s.size() && s[i] == ']') {
        ++i;
        return v;
      }
      fail("expected ',' or ']'");
    }
  }

  JsonPtr stringValue() {
    auto v = std::make_shared<JsonValue>();
    v->type = JsonValue::Type::String;
    ++i; // '"'
    std::string out;
    while (i < s.size() && s[i] != '"') {
      if (s[i] != '\\') {
        out += s[i++];
        continue;
      }
      if (++i >= s.size())
        fail("unterminated escape");
      switch (s[i]) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        // Species and material names are ASCII, so only the Latin-1 range can
        // appear here, written as \u00xx by a JSON writer escaping non-ASCII.
        if (i + 4 >= s.size())
          fail("truncated \\u escape");
        const unsigned code =
            static_cast<unsigned>(std::strtoul(s.substr(i + 1, 4).c_str(),
                                               nullptr, 16));
        if (code > 0xFF)
          fail("non-Latin-1 character in a mechanism file");
        out += static_cast<char>(code);
        i += 4;
        break;
      }
      default:
        fail("unknown escape");
      }
      ++i;
    }
    if (i >= s.size())
      fail("unterminated string");
    ++i; // closing '"'
    v->text = out;
    return v;
  }

  JsonPtr numberValue() {
    auto v = std::make_shared<JsonValue>();
    v->type = JsonValue::Type::Number;
    const char *begin = s.c_str() + i;
    char *end = nullptr;
    v->number = std::strtod(begin, &end);
    if (end == begin)
      fail("expected a number");
    i += static_cast<size_t>(end - begin);
    return v;
  }
};

inline Material materialFromName(const std::string &name) {
  return MaterialMap::fromString(name);
}

} // namespace impl

/// Build a ChemicalMechanism from the text of a mechanism file.
template <typename NumericType>
ChemicalMechanism<NumericType>
parseChemicalMechanism(const std::string &text) {
  using Json = impl::JsonValue;
  const auto root = impl::JsonParser(text).parse();

  const auto version = root->get("schemaVersion");
  if (!version || version->isNull()) {
    VIENNACORE_LOG_ERROR("Mechanism file has no schemaVersion. It predates the "
                         "published format; regenerate it with ViennaChem.");
  } else if (static_cast<int>(version->number) !=
             chemicalMechanismSchemaVersion) {
    VIENNACORE_LOG_ERROR(
        "Mechanism file is schema version " +
        std::to_string(static_cast<int>(version->number)) +
        ", but this ViennaPS reads version " +
        std::to_string(chemicalMechanismSchemaVersion) +
        ". Regenerate it with a matching ViennaChem.");
  }

  ChemicalMechanism<NumericType> mech;

  mech.name = root->str("name");
  mech.temperature =
      static_cast<NumericType>(root->at("constants").num("temperature", 300.));

  // the solid phases, in file order, so a reaction's solidAtoms line up
  for (const auto &s : root->at("solids").array) {
    const int si = mech.addSolid(s->str("name"),
                                 static_cast<NumericType>(s->num("rho", 1.)));
    if (auto perMaterial = s->get("rhoMaterials"))
      for (const auto &[name, rho] : perMaterial->object)
        mech.setSolidDensity(si, impl::materialFromName(name),
                             static_cast<NumericType>(rho->number));
  }

  const auto &siteTypes = root->at("siteTypes");
  mech.setSiteTypeCount(static_cast<int>(siteTypes.array.size()));
  if (!siteTypes.array.empty()) {
    const auto density = siteTypes.array.front()->get("density");
    if (density && !density->isNull())
      mech.siteDensity = static_cast<NumericType>(density->number);
  }

  for (const auto &c : root->at("coverages").array)
    mech.addCoverage(c->str("name"), 0., static_cast<int>(c->num("site", 0.)));

  // the ion source, declared before the gas species so that a yield channel's
  // index follows the file's gas ordering
  if (auto src = root->get("ionSource"); src && !src->isNull()) {
    mech.setIonSource(static_cast<NumericType>(src->num("meanEnergy", 100.)),
                      static_cast<NumericType>(src->num("sigmaEnergy", 10.)),
                      static_cast<NumericType>(src->num("exponent", 200.)));
    mech.setIonReflection(
        static_cast<NumericType>(src->num("inflectAngle", 89.)),
        static_cast<NumericType>(src->num("n_l", 10.)),
        static_cast<NumericType>(src->num("minAngle", 80.)),
        static_cast<NumericType>(src->num("thetaRMin", 70.)),
        static_cast<NumericType>(src->num("thetaRMax", 90.)));
  }

  // A per-material rate constant, in the one uniform form the file always uses.
  const auto readMaterialConstants =
      [](const Json &owner, auto &&setDefault, auto &&setNamed) {
        auto constants = owner.get("materialConstants");
        if (!constants || constants->isNull())
          return;
        const auto &fallback = owner.at("materialDefault");
        setDefault(fallback.num("prefactor", 0.), fallback.num("Ea", 0.),
                   fallback.num("beta", 0.));
        for (const auto &[name, spec] : constants->object)
          setNamed(name, spec->num("prefactor", 0.), spec->num("Ea", 0.),
                   spec->num("beta", 0.));
      };

  for (const auto &g : root->at("gas").array) {
    const auto label = g->get("label");
    const auto flux = g->get("flux");
    const int idx = mech.addGasSpecies(
        (label && !label->isNull()) ? label->text : g->str("name"),
        static_cast<NumericType>((flux && !flux->isNull()) ? flux->number : 0.),
        g->flag("traced"));

    auto sticking = g->get("sticking");
    if (!sticking || sticking->isNull())
      continue;
    mech.setSticking(idx, static_cast<NumericType>(sticking->num("s0", 0.)),
                     static_cast<NumericType>(sticking->num("Ea", 0.)),
                     static_cast<int>(sticking->num("freeSiteExp", 0.)),
                     static_cast<NumericType>(sticking->num("beta", 0.)),
                     static_cast<int>(sticking->num("site", 0.)));
    // the particle re-emits with the sticking of the material it hit
    readMaterialConstants(
        *sticking,
        [&](double pre, double Ea, double beta) {
          mech.setStickingMaterialConstantDefault(
              idx, static_cast<NumericType>(pre), static_cast<NumericType>(Ea),
              static_cast<NumericType>(beta));
        },
        [&](const std::string &name, double pre, double Ea, double beta) {
          mech.setStickingMaterialConstant(
              idx, impl::materialFromName(name), static_cast<NumericType>(pre),
              static_cast<NumericType>(Ea), static_cast<NumericType>(beta));
        });
  }

  int yieldIndex = 0;
  for (const auto &rx : root->at("reactions").array) {
    const auto &constant = rx->at("constant");
    const bool isAdsorption = constant.str("kind") == "sticking";
    const double prefactor =
        isAdsorption ? constant.num("s0", 0.) : constant.num("k0", 0.);

    std::vector<int> freeSiteExponent;
    for (const auto &e : rx->at("freeSiteExp").array)
      freeSiteExponent.push_back(static_cast<int>(e->number));
    std::vector<NumericType> nu;
    for (const auto &n : rx->at("nu").array)
      nu.push_back(static_cast<NumericType>(n->number));
    const auto &solidAtoms = rx->at("solidAtoms").array;

    const int ridx = mech.addReaction(
        static_cast<NumericType>(prefactor),
        static_cast<NumericType>(constant.num("Ea", 0.)), isAdsorption,
        freeSiteExponent, nu,
        solidAtoms.empty() ? NumericType(0.)
                           : static_cast<NumericType>(solidAtoms[0]->number),
        static_cast<NumericType>(constant.num("beta", 0.)));
    mech.reactions[ridx].equation = rx->str("eq");
    // a step forming a solid other than the first says which one
    for (size_t si = 1; si < solidAtoms.size(); ++si)
      if (solidAtoms[si]->number != 0.)
        mech.setSolidAtoms(ridx, static_cast<int>(si),
                           static_cast<NumericType>(solidAtoms[si]->number));

    for (const auto &f : rx->at("gasFactors").array)
      mech.addGasFactor(ridx, static_cast<int>(f->num("idx", 0.)),
                        static_cast<int>(f->num("exp", 1.)));
    for (const auto &f : rx->at("covFactors").array)
      mech.addCoverageFactor(ridx, static_cast<int>(f->num("idx", 0.)),
                             static_cast<int>(f->num("exp", 1.)));

    // an ion-driven step: declare its yield channel and take that channel's
    // flux as an ordinary gas factor, since the yield is folded into it
    if (auto y = rx->get("ionYield"); y && !y->isNull()) {
      const int channel = mech.addIonYield(
          y->str("label"), static_cast<NumericType>(y->num("A", 1.)),
          static_cast<NumericType>(y->num("Eth", 0.)),
          static_cast<NumericType>(y->num("B", 0.)), y->flag("enhanced"),
          static_cast<NumericType>(y->num("flux", 1.)));
      if (auto perMaterial = y->get("materials"))
        for (const auto &[name, spec] : perMaterial->object)
          mech.setIonYieldMaterial(
              yieldIndex, impl::materialFromName(name),
              static_cast<NumericType>(spec->num("A", 0.)),
              static_cast<NumericType>(spec->num("Eth", 0.)));
      ++yieldIndex;
      mech.addGasFactor(ridx, channel, 1);
    }

    readMaterialConstants(
        *rx,
        [&](double pre, double Ea, double beta) {
          mech.setMaterialConstantDefault(ridx, static_cast<NumericType>(pre),
                                          static_cast<NumericType>(Ea),
                                          static_cast<NumericType>(beta));
        },
        [&](const std::string &name, double pre, double Ea, double beta) {
          mech.setMaterialConstant(ridx, impl::materialFromName(name),
                                   static_cast<NumericType>(pre),
                                   static_cast<NumericType>(Ea),
                                   static_cast<NumericType>(beta));
        });
  }

  return mech;
}

/// Build a ChemicalMechanism from a mechanism file on disk.
template <typename NumericType>
ChemicalMechanism<NumericType>
readChemicalMechanism(const std::string &filename) {
  std::ifstream file(filename);
  if (!file)
    VIENNACORE_LOG_ERROR("Could not open mechanism file '" + filename + "'.");
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parseChemicalMechanism<NumericType>(buffer.str());
}

} // namespace viennaps
