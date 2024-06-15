import sympy as sp
from sympy.printing import cxxcode

x,y,z = sp.symbols('x y z', real=True)
e1, e2 = sp.symbols('e1 e2', real=True)

F = (x**(2/e2) + y**(2/e2))**(e2/e1) + z**(2/e1)

p_vars = [x, y, z]
dim_p = 3

s_vars = ['sx', 'sy', 'sz']

# F_dP
for i in range(dim_p):
    result = sp.simplify(F.diff(p_vars[i]))
    print(f'F_dP({i}) = {cxxcode(result)};')
    print(f'F_dP({i}) = {s_vars[i]} * F_dP({i});')

# F_dPdP
# for i in range(dim_p):
#     for j in range(dim_p):
#         result = sp.simplify(F.diff(p_vars[i]).diff(p_vars[j]))
#         if result != 0:
#             print(f'F_dPdP({i},{j}) = {cxxcode(result)};')
#             print(f'F_dPdP({i},{j}) = {s_vars[i]} * {s_vars[j]} * F_dPdP({i},{j});')

# F_dPdPdP
# for i in range(dim_p):
#     for j in range(dim_p):
#         for k in range(dim_p):
#             result = sp.simplify(F.diff(p_vars[i]).diff(p_vars[j]).diff(p_vars[k]))
#             if result != 0:
#                 print(f'F_dPdPdP({i},{j},{k}) = {cxxcode(result)};')
#                 print(f'F_dPdPdP({i},{j},{k}) = {s_vars[i]} * {s_vars[j]} * {s_vars[k]} * F_dPdPdP({i},{j},{k});')