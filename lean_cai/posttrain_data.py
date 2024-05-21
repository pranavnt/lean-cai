def get_dataset():
  data = [
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.Linarith

theorem aopsbook_v1_c7_em3 :
  1 / (3 + 2 * Real.sqrt 2) + 1 / (2 * Real.sqrt 2 + Real.sqrt 7) + 1 / (Real.sqrt 7 + Real.sqrt 6) + 1 / (Real.sqrt 6 + Real.sqrt 5) + 1 / (Real.sqrt 5 + 2) + 1 / (2 + Real.sqrt 3) = 1 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_em3 (x : ℝ) (h₀ : x^2 + 3 / 2 * x = 1) : x = -2 ∨ x = 1 / 2 := by
  sorry""",
  """import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Modeq

theorem aopsbook_v1_c28_p516 (n : ℕ) :
  Finset.card (Finset.filter (fun x => 3∣x) (Finset.range (7 * (n + 1)) \ Finset.range (7 * n + 1))) = 2 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p567 (x y : ℝ) (h₀ : x ≠ 0 ∧ y ≠ 0) (h₁ : 2 / x = y / 3) (h₂ : y / 3 = x / y) : x^3 = 12 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c3_p52 (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : (x + 2*y) / (x*y) = 11/12)
    (h₃ : (2*x - 3*y) / (x*y) = 2/3) : x = 6 ∧ y = 12/7 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Sqrt

theorem aopsbook_v1_c29_p530 (x : ℝ) (h₀ : 0 ≤ 5 - x) (h₁ : Real.sqrt (5 - x) = x * Real.sqrt (5 - x)) :
    x = 1 ∨ x = 5 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c22_p423 (x y : ℝ) (h₀ : x < x - y) (h₁ : x + y < y) : x < 0 ∧ y < 0 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt

theorem aopsbook_v1_c29_p571 (a b c x y z : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x + y + z ≠ 0)
    (h₁ : x / a = y / b) (h₂ : y / b = z / c) (h₃ : z / c = (x * y * z) / (x + y + z)) :
    x = (a + b + c) / (b * c) := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Sqrt

theorem aopsbook_v1_c6_ex11_2 : Real.sqrt (55 - 10 * Real.sqrt 10) = 5 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c22_p415 (z : ℝ) (h₀ : z ≠ 0) (h₁ : 1 ≤ 2 / z) : 0 < z ∧ z ≤ 2 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_p137 (a b : ℝ) : (a + b) / a ≠ b / (a + b) := by
  sorry""",
  """import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex4_1 (x : ℝ) (h₀ : 3 * x^2 + 5 * x = 0) : x = -5 / 3 ∨ x = 0 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Sqrt

theorem aopsbook_v2_c13_p191 :
    (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / (Real.sqrt (Real.sqrt 5 + 1)) -
    Real.sqrt (3 - 2 * Real.sqrt 2) = 1 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c13_p187 (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1) : -1 / 2 ≤ a * b + b * c + c * a := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Int.Modeq

theorem aopsbook_v1_c28_p524 (n : ℤ) : 10∣n^5 - n := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic

theorem aopsbook_v1_c7_p161 (q : ℤ) (h₀ : ∃ x y, q = x^2 + y^2) :
    (∃ m n, 2 * q = m^2 + n^2) ∧ (∃ c d, 5 * q = c^2 + d^2) := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c4_p86 (x y z : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : z > 0)
    (h₀ : x/y = y/(x-z)) (h₁ : x/y = (x+y)/z) : x/y = 2 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c22_p425 (x : ℝ) (h₀ : x ≠ 0) (h₁ : x + 1 / x ≤ -2) : x < 0 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Sqrt

theorem aopsbook_v2_c13_intro5 : Real.sqrt (6 + Real.sqrt 11) - Real.sqrt (6 - Real.sqrt 11) = Real.sqrt 2 := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p576 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) (h₁ : ∃ x, x^2 + a * x + 2 * b = 0)
    (h₁ : ∃ x, x^2 + 2 *b * x + a = 0) : 6 ≤ a + b := by
  sorry""",
  """import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c14_p221 (a : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^3 - 12 * x^2 + a * x -64)
    (h₁ : ∃ x ≥ 0, f x = 0) : a = 48 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex2_1
  (x : ℝ)
  (h₀ : x^2 = -5 * x - 6) :
  x = -2 ∨ x = -3 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c29_p556
  (x : ℝ)
  (h₀ : x = Real.sqrt (2 + Real.sqrt 2) - Real.sqrt (2 - Real.sqrt 2)) :
  384 * x^2 - x^8 = 448 := by
  sorry""",
  """import Mathlib.Data.Complex.Basic

theorem aopsbook_v1_c6_em10
  (x y : ℂ)
  (h₀ : x + y = x * y)
  (h₁ : x * y = 2) :
  (x = 1 - Complex.I ∧ y = 1 + Complex.I) ∨ (x = 1 + Complex.I ∧ y = 1 - Complex.I) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_em8
  (y : ℝ)
  (h₀ : y ≠ 2 ∧ y ≠ 6)
  (h₁ : 1 + (y + 3) / (y - 2) = (3 * y - 3) / (6 - y)) :
  y = 0 ∨ y = 4 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c6_p132 :
  Real.sqrt (53 - 8 * Real.sqrt 15) = 4 * Real.sqrt 3 - Real.sqrt 5 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex4_4
  (x : ℝ)
  (h₀ : x^2 / 3 - 2 * x + 3 = 0) :
  x = 3 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c13_p209_imo1985
  (a b c : ℝ)
  (h₀ : b * c - a^2 ≠ 0)
  (h₁ : c * a - b^2 ≠ 0)
  (h₂ : a * b - c^2 ≠ 0)
  (h₃ : (b * c - a^2)⁻¹ + (c * a - b^2)⁻¹ + (a * b - c^2)⁻¹ = 0) :
  a * ((b * c - a^2)⁻¹)^2 + b * ((c * a - b^2)⁻¹)^2 + c * ((a * b - c^2)⁻¹)^2 = 0 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c29_p580
  (a b p : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = x^2 + p * x + 8)
  (h₁ : f a = 0)
  (h₂ : f b = 0)
  (h₃ : a ≠ b) :
  4 * Real.sqrt 2 < abs (a + b) := by
  sorry""",
  """import Mathlib.Tactic.NormNum

theorem aopsbook_v1_c7_p138
  (h₀ : 9876^2 = 97535376) :
  9877^2 = 97555129 := by
  norm_num""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex2_2
  (x : ℝ)
  (h₀ : x^2 - 3 * x - 40 = 0) :
  x = 8 ∨ x = -5 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v1_c3_p37
  (a b c : ℝ)
  (h₀ : c ≠ 0)
  (h₁ : a ≠ 0)
  (h₂ : 0 ≤ b / c)
  (h₃ : 0 ≤ a + (b / c))
  (h₄ : Real.sqrt (a + (b / c)) = a * Real.sqrt (b / c)) :
  c = b * (a^2 - 1) / a := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v1_c7_ex4
  (a b : ℝ)
  (h₀ : a * b = 4)
  (h₁ : a^2 + b^2 = 4) :
  a^3 + b^3 = 0 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_em13
  (x : ℝ)
  (h₀ : x^4 + 3 * x^2 - 4 = 0) :
  x = -1 ∨ x = 1 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem aopsbook_v1_c11_p191
  (a b x : ℝ)
  (h₀ : 0 < b ∧ b < a)
  (h₁ : 0 < x ∧ x < π / 2)
  (h₂ : Real.tan x = 2 * a * b / (a^2 - b^2)) :
  Real.sin x = 2 * a * b / (a^2 + b^2) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c28_p512
  (x y : ℝ)
  (h₀ : x ≠ 0 ∧ y ≠ 0)
  (h₁ : ∃ n : ℤ, x / y = ↑n)
  (h₁ : ∃ m : ℤ, y / x = ↑m) :
  abs x = abs y := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c7_p159
  (a b : ℝ)
  (h₀ : a^3 - b^3 = 24)
  (h₁ : a - b = 2) :
  a + b = 2 * Real.sqrt 33 / 3 ∨ a + b = -2 * Real.sqrt 33 / 3 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_p112
  (x : ℝ)
  (h₀ : x ≠ -1 ∧ x ≠ 2) :
  (2 * x^2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Polynomials

theorem aopsbook_v1_c28_ex3
  (a b c p : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f x = a * x^2 + b * x + c)
  (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) :
  f p = 0 → p < 0 := by
  sorry""",
  """import Mathlib.Data.Complex.Basic

theorem aopsbook_v1_c6_p113
  (z : ℂ)
  (h₀ : z ≠ 0 ∧ z ≠ 1)
  (h₁ : z / (z - 1) = (z + 1) / z - 2) :
  z = (1 + Complex.I) / 2 ∨ z = (1 - Complex.I) / 2 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c7_em6
  (a b : ℝ)
  (h₀ : a * b = 3)
  (h₁ : a + b = 6) :
  1 / a + 1 / b = 2 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c7_p158
  (a b : ℝ)
  (h₀ : a^4 + b^4 = 16)
  (h₁ : a + b = 2) :
  a * b = 0 ∨ a * b = 8 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c6_ex9
  (z : ℝ)
  (h₀ : 0 ≤ z ∧ z ≤ 1)
  (h₁ : Real.sqrt (5 * z + 5) - Real.sqrt (3 - 3 * z) - 2 * Real.sqrt z = 0) :
  z = 1 / 4 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem aopsbook_v2_c14_em7
  (α β : ℝ)
  (h₀ : 0 < α ∧ α < π / 2)
  (h₁ : 0 < β ∧ β < π / 2) :
  1 ≤ ((Real.cos α)^3 / Real.cos β + (Real.sin α)^3 / Real.sin β) * Real.cos (α - β) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c13_p199
  (a b c : ℝ)
  (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h₁ : (a + b - c) / c = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (-a + b + c) / a) :
  ((a + b) * (b + c) * (c + a) / (a * b * c)) = 8 ∨ ((a + b) * (b + c) * (c + a) / (a * b * c)) = -1 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem aopsbook_v1_c7_p154
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : x^2 + 1 / x^2 = 7) :
  x^3 + 1 / x^3 = 18 ∨ x^3 + 1 / x^3 = -18 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c6_em1
  (x : ℝ)
  (h₀ : 2*x + 1 ≠ 0) :
  (8*x^4 - 12*x^3 + 2*x + 1) / (2*x + 1) = 4*x^3 - 8*x^2 + 4*x - 1 + 2 / (2*x + 1) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c6_em1
  (x : ℝ)
  (h₀ : 2*x + 1 ≠ 0) :
  (8*x^4 - 12*x^3 + 2*x + 1) / (2*x + 1) = 4*x^3 - 8*x^2 + 4*x - 1 + 2 / (2*x + 1) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c28_p509
  (x : ℝ)
  (h₀ : x^5 + 3 * x^2 + 7 * x + 2 = 0) :
  x < 0 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex2_3
  (x : ℝ)
  (h₀ : 2 * x^2 + 1 / 3 * x - 2 / 3 = 0) :
  x = -2 / 3 ∨ x = 1 / 2 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c14_ex4
  (a b : ℝ)
  (h₀ : 0 < a ∧ 0 < b) :
  2 ≤ a / b + b / a := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex4_5
  (x : ℝ)
  (h₀ : 4 * x^2 = 5 * x) :
  x = 0 ∨ x = 5 / 4 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c13_p204
  (x y z : ℝ)
  (h₀ : x * y * z = 4)
  (h₁ : x^3 + y^3 + z^3 = 4)
  (h₂ : x * y^2 + x^2 * y + y * z^2 + y^2 * z + z * x^2 + z^2 * x = 12) :
  x * y + y * z + z * x = 6 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c14_intro2
  (x y z : ℝ)
  (h₀ : 0 < x ∧ 0 < y ∧ 0 < z)
  (h₁ : 2 * (x * y + y * z + z * x) = 96) :
  x * y * z ≤ 64 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p549
  (x y : ℝ)
  (h₀ : x + y = 7)
  (h₁ : x^2 - y^2 = 21) :
  2 * x + 3 * y = 16 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c28_p514
  (x : ℝ)
  (h₀ : 0 < abs x + x) :
  0 < x := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex4_2
  (x : ℝ)
  (h₀ : 3 * x^2 + 6 * x + 3 = 0) :
  x = -1 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c13_p192
  (x y z : ℝ)
  (h₀ : x + y - z = 0)
  (h₁ : z * x - x * y + y * z = 27)
  (h₂ : x * y * z = 54) :
  (x, y, z) = (-6, 3, -3) ∨ (x, y, z) = (3, -6, -3) ∨ (x, y, z) = (3, 3, 6) := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem aopsbook_v2_c14_p212
  (α β : ℝ)
  (h₀ : 0 < α ∧ α < π / 2)
  (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : ((Real.cos α)^3 / Real.cos β + (Real.sin α)^3 / Real.sin β) * Real.cos (α - β) = 1) :
  α = β := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex2_4
  (x : ℝ)
  (h₀ : 49 * x^2 - 316 * x + 132 = 0) :
  x = 6 ∨ x = 22 / 49 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c22_p417
  (t : ℝ) :
  -t^2 + 60 * t + 700 ≤ 1600 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c6_p118
  (x : ℝ)
  (h₀ : 0 ≤ 40 - 9 * x ∧ 0 ≤ 7 - x ∧ 0 ≤ -x)
  (h₁ : Real.sqrt (40 - 9 * x) - 2 * Real.sqrt (7 - x) = Real.sqrt (-x)) :
  2 * x + 5 = -13 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c7_p153 :
  1 / (Real.sqrt 15 + Real.sqrt 13) + 1 / (Real.sqrt 13 + Real.sqrt 11) + 1 / (Real.sqrt 11 + 3) + 1 / (3 + Real.sqrt 7) + 1 / (Real.sqrt 7 + Real.sqrt 5) = (Real.sqrt 15 - Real.sqrt 5) / 2 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c7_p152
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : Real.sqrt x + 1 = x - Real.sqrt x - 1) :
  x = 4 + 2 * Real.sqrt 3 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c7_ex3
  (z : ℝ)
  (h₀ : 0 < z)
  (h₁ : z^2 + 1 / z^2 = 14) :
  z^5 + 1 / z^5 = 724 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c14_p229
  (a b c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : a + b + c = 6) :
  75 / 4 ≤ (a + 1 /b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c6_ex11_1 :
  Real.sqrt (35 - 10 * Real.sqrt 10) = 5 - Real.sqrt 10 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c6_ex2_5
  (x : ℝ)
  (h₀ : x ≠ 3)
  (h₁ : x = 28 / (x - 3)) :
  x = 7 ∨ x = -4 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c22_p416
  (x : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 0 ≤ x^2 + x - 30)
  (h₂ : 0 < 6 / x) :
  5 ≤ x := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c14_p213
  (a b c d : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h₁ : a + 2 * b + 3 * c + 4 * d = 8) :
  a * b * c * d ≤ 2 / 3 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c14_ex2
  (x y : ℝ) :
  (x * y + 1)^2 ≤ (x^2 + 1) * (y^2 + 1) := by
  sorry""",
  """import Mathlib.Analysis.SpecialFunctions.Sqrt

theorem aopsbook_v1_c6_ex10
  (x : ℝ)
  (h₀ : Real.sqrt (x^2 + 1) + x^2 + 1 = 90) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v2_c13_p202
  (a b c : ℝ)
  (h₀ : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2 * a * b) / (a^2 + c^2 - b^2 + 2 * a * c) = (a + b - c) / (a - b + c) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p548
  (x y z : ℝ)
  (h₀ : y ≠ 0 ∧ x ≠ 0)
  (h₁ : x / y = 4 * y / x)
  (h₂ : 4 * y / x = z) :
  z = 2 ∨ z = -2 := by
  sorry""",
  """import Mathlib.Analysis.Calculus.MeanValue

theorem aopsbook_v2_c13_ex9
  (a b c d : ℝ) :
  (a + b + c + d)^3 = a^3 + b^3 + c^3 + d^3 + 3 * (a^2 * b + b^2 * a + a^2 * c + c^2 * a + a^2 * d + d^2 * a + b^2 * c + c^2 * b + b^2 * d + d^2 * b + c^2 * d + d^2 * c) + 6 * (a * b * c + a * b * d + a * c * d + b * c * d) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p533
  (r : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ n, f n = 1 / 3 * n * (n + 1) * (n + 2)) :
  f r - f (r - 1) = r * (r + 1) := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem aopsbook_v1_c29_p572
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : a ≠ 0)
  (h₁ : ∀ x, f x = a * x^2 + 2 * b * x + c)
  (h₂ : ∀ x y, f x = 0 ∧ f y = 0 → x = y) :
  a * c = b^2 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem amc12a_2009_p9
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f (x + 3) = 3 * x^2 + 7 * x + 4)
  (h₁ : ∀ x, f x = a * x^2 + b * x + c) :
  a + b + c = 2 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_13
  (a b : ℝ)
  (h₀ : ∀ x, (x - 3 ≠ 0 ∧ x - 5 ≠ 0) → 4 * x / (x^2 - 8 * x + 15) = a / (x - 3) + b / (x - 5)) :
  a = -6 ∧ b = 10 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem imo_2006_p6
  (a b c : ℝ) :
  (a * b * (a^2 - b^2)) + (b * c * (b^2 - c^2)) + (c * a * (c^2 - a^2)) ≤ (9 * Real.sqrt 2) / 32 * (a^2 + b^2 + c^2)^2 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem imo_1965_p1
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 2 * Real.pi)
  (h₂ : 2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))))
  (h₃ : abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem amc12a_2009_p5
  (x : ℝ)
  (h₀ : x^3 - (x + 1) * (x - 1) * x = 5) :
  x^3 = 125 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Complex.Basic

theorem mathd_algebra_73
  (p q r x : ℂ)
  (h₀ : (x - p) * (x - q) = (r - p) * (r - q))
  (h₁ : x ≠ r) :
  x = p + q - r := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

theorem imo_1962_p4
  (S : Set ℝ)
  (h₀ : S = {x : ℝ | (Real.cos x)^2 + (Real.cos (2 * x))^2 + (Real.cos (3 * x))^2 = 1}) :
  S = {x : ℝ | ∃ m : ℤ, (x = π / 2 + m * π) ∨ (x = π / 4 + m * π / 2) ∨ (x = π / 6 + m * π) ∨ (x = 5 * π / 6 + m * π)} := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs

theorem mathd_numbertheory_236 :
  (1999^2000) % 5 = 1 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_101
  (x : ℝ)
  (h₀ : x^2 - 5 * x - 4 ≤ 10) :
  x ≥ -2 ∧ x ≤ 7 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem amc12_2000_p5
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs

theorem mathd_numbertheory_200 :
  139 % 11 = 7 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_140
  (a b c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : ∀ x, 24 * x^2 - 19 * x - 35 = (((a * x) - 5) * ((2 * (b * x)) + c))) :
  a * b - 3 * c = -9 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_455
  (x : ℝ)
  (h₀ : 2 * (2 * (2 * (2 * x))) = 48) :
  x = 3 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs

theorem mathd_numbertheory_45 :
  (Nat.gcd 6432 132) + 11 = 23 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs

theorem mathd_numbertheory_739 :
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) % 10 = 0 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_245
  (x : ℝ)
  (h₀ : x ≠ 0) :
  (4 / x)⁻¹ * ((3 * x^3) / x)^2 * ((1 / (2 * x))⁻¹)^3 = 18 * x^8 := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Real.Basic

theorem mathd_algebra_28
  (c : ℝ) :
  3 * (c + 3) - 5 * c = 15 - 2 * c := by
  sorry""",
  """import Mathlib.Algebra.Group.Defs
import Mathlib.Data.Nat.Prime

theorem mathd_numbertheory_332 :
  3 * 5 * 7 = 105 := by
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem mathd_algebra_478
  (b h v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2) :
  v = 65 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_141
  (a b : ℝ)
  (h₁ : (a * b)=180)
  (h₂ : 2 * (a + b)=54) :
  (a^2 + b^2) = 369 :=
  sorry""",
  """import Mathlib.Data.Real.Basic

theorem imo_1983_p6
  (a b c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : c < a + b)
  (h₂ : b < a + c)
  (h₃ : a < b + c) :
  0 ≤ a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_33
  (x y z : ℝ)
  (h₀ : x ≠ 0)
  (h₁ : 2 * x = 5 * y)
  (h₂ : 7 * y = 10 * z) :
  z / x = 7 / 25 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_numbertheory_299 :
  (1 * 3 * 5 * 7 * 9 * 11 * 13) % 10 = 5 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem amc12b_2020_p2 :
  ((100 ^ 2 - 7 ^ 2):ℝ) / (70 ^ 2 - 11 ^ 2) * ((70 - 11) * (70 + 11) / ((100 - 7) * (100 + 7))) = 1 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_419
  (a b : ℝ)
  (h₀ : a = -1)
  (h₁ : b = 5) :
  -a - b^2 + 3 * (a * b) = -39 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_398
  (a b c : ℝ)
  (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₁ : 9 * b = 20 * c)
  (h₂ : 7 * a = 4 * b) :
  63 * a = 80 * c :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_137
  (x : ℕ)
  (h₀ : ↑x + (4:ℝ) / (100:ℝ) * ↑x = 598) :
  x = 575 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_160
  (n x : ℝ)
  (h₀ : n + x = 97)
  (h₁ : n + 5 * x = 265) :
  n + 2 * x = 139 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_24
  (x : ℝ)
  (h₀ : x / 50 = 40) :
  x = 2000 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_176
  (x : ℝ) :
  (x + 1)^2 * x = x^3 + 2 * x^2 + x :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_156
  (x y : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀t, f t = t^4)
  (h₁ : ∀t, g t = 5 * t^2 - 6)
  (h₂ : f x = g x)
  (h₃ : f y = g y)
  (h₄ : x^2 < y^2) :
  y^2 - x^2 = 1 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_numbertheory_345 :
  (2000 + 2001 + 2002 + 2003 + 2004 + 2005 + 2006) % 7 = 0 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_numbertheory_328 :
  (5^999999) % 7 = 6 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_algebra_452
  (a : ℕ → ℝ)
  (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
  (h₁ : a 1 = 2 / 3)
  (h₂ : a 9 = 4 / 5) :
  a 5 = 11 / 15 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1993_p9
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f (2 * x) = 2 / f x)
  (h₁ : f 2 = 1) :
  f 12 = 1 / 32 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1984_p10
  (a b c : ℝ)
  (h₀ : 1 + 3 * a = 2 * b)
  (h₁ : 2 * (3 + a) = 9 + c) :
  3 + a + b + c = 5 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1984_p1
  (f : ℝ → ℝ)
  (h₀ : ∀ x, f (x^3) = (f x)^3)
  (h₁ : f (f 2) = 4) :
  f (-2) = -2 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1994_p1
  (a b : ℝ)
  (h₀ : a + b = 8)
  (h₁ : a * b = 6) :
  a^3 + b^3 = 392 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1986_p6
  (a b : ℝ)
  (h₀ : a^3 + 8 * b^3 = 4)
  (h₁ : a * b = 1 / 2) :
  a + 2 * b = 1 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1985_p6
  (x y : ℕ)
  (h₀ : y = 2 * x + 1)
  (h₁ : x^2 + y^2 = 6565) :
  x = 54 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1985_p8
  (x : ℕ)
  (h₀ : 10000 * x = 6 * (10^5)) :
  x = 6 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem aime_1992_p8
  (a b : ℕ)
  (h₀ : 0 < a ∧ a < b)
  (h₁ : (a^2 + b^2)∣(a^2 * b^2)) :
  a = 1 :=
  sorry""",
  """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem mathd_numbertheory_1124
  (n : ℕ)
  (h₀ : n ≤ 9)
  (h₁ : 18∣374 * 10 + n) :
  n = 4 :=
  sorry"""
  ]

  return [F2FData(t) for t in data]

class F2FData:
  def __init__(self, theorem: str):
    self.theorem = theorem

  def train_str(self) -> str:
    return self.theorem.replace("sorry", "")


print(get_dataset()[20].train_str())
