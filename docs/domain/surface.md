---
layout: default
title: Surface and Material Interfaces
parent: Simulation Domain
nav_order: 0
---
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>


# Surface and Material Interfaces
{: .fs-9 .fw-500 }

---

The surface, as well as the material interfaces, are described implicitly by a level set (LS) function $\phi(\vec{x})$ which is defined at every point $\vec{x}$ in space. 
This function is obtained using signed distance transforms, describing the surface $S$ as the zero LS:

$$ 
S = \{\vec{x}\!: \, \phi(\vec{x}) = 0 \}. 
$$

If the domain contains multiple LSs, the top LS wraps the entire structure and therefore represents the surface, while all other LS functions just describe material interfaces. Formally, the different material regions can be described by ${M}$ LS functions satisfying

$$ 
\Phi_k(\vec{x}) \leq 0 \quad \Leftrightarrow \quad \vec{x} \in \bigcup_{i=1}^k \mathcal{M}_i.
$$

Here $\Phi_M$ describes the entire structure $\mathcal{M}$, and the other LS functions correspond to material interfaces. 

When inserting a new LS into the domain, an automatic wrapping process ensues. This process involves enveloping all existing Level-Sets through a Boolean operation, specifically a union with the topmost LS. It's worth noting, though, that this default behavior is not obligatory. In instances where a specialized domain structure is desired, users have the option to circumvent this automatic wrapping mechanism. 

---

![]({% link assets/images/fullGrid.png %})
