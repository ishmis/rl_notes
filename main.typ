#import "@preview/summy:0.1.0": *

#show: cheatsheet.with(
  title: "Cheatsheet Title", 
  authors: "Authors",
  write-title: false,
  font-size: 6.5pt,
  line-skip: 5.5pt,
  x-margin: 10pt,
  y-margin: 10pt,
  num-columns: 4,
  column-gutter: 3pt,
  numbered-units: false,
)

#include "units/intro.typ"
#include "units/mab.typ"
#include "units/mdp.typ"
#include "units/dp.typ"
#include "units/mc.typ"
#include "units/td.typ"
#include "units/planning_and_learning.typ"
#include "units/value_approx.typ"
#include "units/policy_gradient.typ"
#include "units/faq.typ"