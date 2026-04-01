import viennaps as ps


def test_material_registry():
    registry = ps.MaterialRegistry.instance()

    built_in = registry.registerMaterial("Si")
    assert built_in.isBuiltIn()
    assert built_in == ps.Material.Si
    assert registry.customMaterialCount() == 0

    custom_a = registry.registerMaterial("CustomFoo")
    custom_a_repeat = registry.registerMaterial("CustomFoo")
    custom_b = registry.registerMaterial("CustomBar")
    assert registry.customMaterialCount() == 2

    assert custom_a.isCustom()
    assert custom_a == custom_a_repeat
    assert custom_a != custom_b

    domain = ps.Domain(10.0, 10.0, 1.0)
    ps.MakePlane(domain).apply()
    domain.duplicateTopLevelSet("CustomMaterial")

    customMaterial = ps.MaterialMap.fromString("CustomMaterial")
    assert customMaterial == ps.MaterialMap.fromString("CustomMaterial")
    assert customMaterial != ps.MaterialMap.fromString("Si")


if __name__ == "__main__":
    test_material_registry()
