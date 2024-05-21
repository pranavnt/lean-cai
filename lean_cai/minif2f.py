# class MiniF2f:
#   def __init__(self):
#     self.imports = [
#       "algebra.algebra.basic",
#       "algebra.big_operators.basic",
#       "algebra.floor",
#       "algebra.group_power.basic",
#       "algebra.quadratic_discriminant",
#       "algebra.ring.basic",
#       "analysis.asymptotics.asymptotic_equivalent",
#       "analysis.mean_inequalities",
#       "analysis.normed_space.basic",
#       "analysis.inner_product_space.basic",
#       "analysis.inner_product_space.euclidean_dist",
#       "analysis.normed_space.pi_Lp",
#       "analysis.special_functions.exp_log",
#       "analysis.special_functions.pow",
#       "analysis.special_functions.trigonometric.basic",
#       "combinatorics.simple_graph.basic",
#       "data.complex.basic",
#       "data.complex.exponential",
#       "data.equiv.basic",
#       "data.finset.basic",
#       "data.int.basic",
#       "data.int.gcd",
#       "data.int.modeq",
#       "data.list.palindrome",
#       "data.multiset.basic",
#       "data.nat.basic",
#       "data.nat.choose.basic",
#       "data.nat.digits",
#       "data.nat.factorial.basic",
#       "data.nat.fib",
#       "data.nat.modeq",
#       "data.nat.parity",
#       "data.nat.prime",
#       "data.pnat.basic",
#       "data.pnat.prime",
#       "data.polynomial",
#       "data.polynomial.basic",
#       "data.polynomial.eval",
#       "data.rat.basic",
#       "data.real.basic",
#       "data.real.ennreal",
#       "data.real.irrational",
#       "data.real.nnreal",
#       "data.real.sqrt",
#       "data.real.golden_ratio",
#       "data.sym.sym2",
#       "data.zmod.basic",
#       "geometry.euclidean.basic",
#       "geometry.euclidean.circumcenter",
#       "geometry.euclidean.monge_point",
#       "geometry.euclidean.sphere",
#       "init.data.nat.gcd",
#       "linear_algebra.affine_space.affine_map",
#       "linear_algebra.affine_space.independent",
#       "linear_algebra.affine_space.ordered",
#       "linear_algebra.finite_dimensional",
#       "measure_theory.integral.interval_integral",
#       "number_theory.arithmetic_function",
#       "order.bounds",
#       "order.filter.basic",
#       "topology.basic",
#       "topology.instances.nnreal",
#     ]



class LeanFile:
    def __init__(self):
        self.sections = []

    def add_section(self, open_locales, theorem):
        self.sections.append({
            "open_locales": open_locales,
            "theorem": theorem
        })

    def add_theorem(self, name, variables, hypotheses, conclusion):
        theorem = {
            "name": name,
            "variables": variables,
            "hypotheses": hypotheses,
            "conclusion": conclusion
        }
        self.add_section([], theorem)

    def add_theorem_with_open_locales(self, name, open_locales, variables, hypotheses, conclusion):
        theorem = {
            "name": name,
            "variables": variables,
            "hypotheses": hypotheses,
            "conclusion": conclusion
        }
        self.add_section(open_locales, theorem)

    def generate_lean4_file(self, section):
        lean4_file = ""
        lean4_file += "universe u\n"
        lean4_file += "\n"
        lean4_file += "import data.real.basic\n"
        lean4_file += "\n"
        lean4_file += "".join(f"open {locale}\n" for locale in section["open_locales"])
        lean4_file += "\n"
        lean4_file += f"theorem {section['theorem']['name']} ({', '.join(section['theorem']['variables'])}) :\n"
        lean4_file += "  " + section['theorem']['conclusion'] + " :=\n"
        lean4_file += "by\n"
        lean4_file += "  sorry\n"
        return lean4_file

    def get_lean4_files(self):
        lean4_files = []
        for section in self.sections:
            lean4_files.append(self.generate_lean4_file(section))
        return lean4_files

import re

def parse_lean_file(file_path):
    lean_file = LeanFile()

    with open(file_path, "r") as file:
        content = file.read()

    # Find all theorem definitions
    theorem_patterns = r"theorem\s+(\w+)\s*(\(.*?\))?\s*:\s*(.*?)\s*:=\s*begin"
    theorems = re.findall(theorem_patterns, content, re.DOTALL)

    for theorem in theorems:
        name = theorem[0]
        variables = [var.strip() for var in theorem[1].strip("()").split(",")] if theorem[1] else []
        conclusion = theorem[2].strip()

        # Find the open_locales and hypotheses
        open_locales_pattern = r"open_locale\s+(.*?)\n"
        open_locales = re.findall(open_locales_pattern, content)

        hypotheses_pattern = r"begin\n(.*?)\nsorry"
        hypotheses_match = re.search(hypotheses_pattern, content, re.DOTALL)
        hypotheses = [h.strip() for h in hypotheses_match.group(1).split("\n")] if hypotheses_match else []

        if open_locales:
            lean_file.add_theorem_with_open_locales(name, open_locales, variables, hypotheses, conclusion)
        else:
            lean_file.add_theorem(name, variables, hypotheses, conclusion)

    return lean_file

# Example usage
lean_file = parse_lean_file("./data/aops.lean")

lean4_files = lean_file.get_lean4_files()

for lean4_file in lean4_files:
    print(lean4_file)
    print("---")
