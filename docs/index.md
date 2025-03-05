---
title: Home
layout: default
nav_order: 1
---

# ViennaPS
{: .fs-10 }

Process Simulation Library
{: .fs-6 .fw-300 }

[Get started now]({% link inst/index.md %}){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View it on GitHub](https://github.com/ViennaTools/ViennaPS){: .btn .fs-5 .mb-4 .mb-md-0 }

---

ViennaPS is a header-only C++ library for simulating microelectronic fabrication processes. It combines surface and volume representations with advanced level-set methods and Monte Carlo flux calculations, powered by high-performance ray tracing. Users can develop custom models, use pre-configured physical models, or leverage emulation for flexible and efficient process simulations.

ViennaPS is designed to be easily integrated into existing C++ projects and provides Python bindings for seamless use in Python environments. The library is under active development and is continuously improved to meet the evolving needs of process simulation in microelectronics.

{: .note }
> ViennaPS is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know on [GitHub][ViennaPS issues] or contact us directly at [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)!

This documentation serves as your comprehensive guide to understanding, implementing, and harnessing the capabilities of our process simulation library. Whether you are a seasoned researcher seeking to refine your simulations or an engineer aiming to optimize real-world processes, this library provides a versatile and robust platform to meet your diverse needs.

Throughout this documentation, you will find detailed explanations, practical examples, and best practices to effectively utilize the library. We aim to empower users with the knowledge and tools necessary to simulate a wide range of processes accurately and efficiently, making informed decisions and driving innovation in the field.

## Contributing

If you want to contribute to ViennaPS, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html).

Make sure to format all files before creating a pull request:
```bash
cmake -B build
cmake --build build --target format
```

## About the project

ViennaPS was developed under the aegis of the [Institute for Microelectronics](http://www.iue.tuwien.ac.at/) at the __TU Wien__. 

Current contributors: Tobias Reiter, Noah Karnel, Roman Kostal, Lado Filipovic

Contact us via: [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)

## License 

See file [LICENSE](https://github.com/ViennaTools/ViennaPS/blob/master/LICENSE) in the base directory.

[ViennaPS repo]: https://github.com/ViennaTools/ViennaPS
[ViennaPS issues]: https://github.com/ViennaTools/ViennaPS/issues

