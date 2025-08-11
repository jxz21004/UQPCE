from pyxdsm.XDSM import (
    XDSM,
    OPT,
    FUNC,
    SOLVER
)

def generate_xdsm():
    xdsm = XDSM()

    xdsm.add_system('opt', OPT, r'\textbf{Optimizer}')
    xdsm.add_system('solver', SOLVER, r'\textbf{Newton Solver}')


    xdsm.add_system('D1', FUNC, [r'\textbf{Discipline 1}', 
                                 r'y_1 = z_1^2 a_0^\dagger + z_2 a_1^\dagger + x a_2^\dagger - 0.2 y_2'])
    xdsm.add_system('D2', FUNC, [r'\textbf{Discipline 2}',
                                 r'y_2 = \sqrt{y_1} + z_1 a_0^\dagger + z_2 a_1^\dagger'])
    xdsm.add_system('F', FUNC, [r'\textbf{Objective}',
                                r'f = x^2 + z_2 a_1^\dagger + y_1/a_3^\dagger + e^{-y_2} a_0^\dagger'])
    xdsm.add_system('G1', FUNC, [r'\textbf{Constraint 1}',
                                 r'g_1 = 8 - y_1'])
    xdsm.add_system('G2', FUNC, [r'\textbf{Constraint 2}',
                                 r'g_2 = y_2 - 12'])
    xdsm.add_system('UQ', FUNC, [r'\textbf{UQPCE}'])

    # Design variables
    xdsm.add_input('opt', r'x, z_1, z_2')
    
    # Uncertain parameters
    xdsm.add_input('D1', r'a_0^\dagger, a_1^\dagger, a_2^\dagger')
    xdsm.add_input('D2', r'a_0^\dagger, a_1^\dagger')
    xdsm.add_input('F', r'a_0^\dagger, a_1^\dagger, a_3^\dagger')

    # Optimizer to disciplines
    xdsm.connect('opt', 'D1', r'x, z_1, z_2')
    xdsm.connect('opt', 'D2', r'z_1, z_2')

    # Connect variables to solver
    xdsm.connect('D1', 'solver', r'\mathcal{R}({y_1^\dagger})')
    xdsm.connect('D2', 'solver', r'\mathcal{R}({y_2^\dagger})')
    xdsm.connect('solver', 'D1', r'x, z_1, z_2')
    xdsm.connect('solver', 'D2', r'z_1, z_2')


    # Coupling variables
    xdsm.connect('D1', 'D2', r'y_1^\dagger')
    xdsm.connect('D2', 'D1', r'y_2^\dagger')

    # Disciplines to functions
    xdsm.connect('D1', 'F', r'y_1^\dagger')
    xdsm.connect('D2', 'F', r'y_2^\dagger')
    xdsm.connect('D1', 'G1', r'y_1^\dagger')
    xdsm.connect('D2', 'G2', r'y_2^\dagger')

    # Design variables to objective
    xdsm.connect('opt', 'F', r'x, z_1, z_2')

    # Functions to UQPCE
    xdsm.connect('F', 'UQ', r'f^\dagger')
    xdsm.connect('G1', 'UQ', r'g_1^\dagger')
    xdsm.connect('G2', 'UQ', r'g_2^\dagger')

    # UQPCE statistics back to optimizer
    xdsm.connect('UQ', 'opt', r'f_\mu, g_{1_\mu}, g_{2_\mu}')

    # Outputs
    xdsm.add_output('opt', r'x^*, z_1^*, z_2^*', side='right')
    xdsm.add_output('UQ', r'f_{CI}', side='right')

    # Write the XDSM
    xdsm.write("./sellar_uqpce_xdsm")


if __name__ == "__main__":
    generate_xdsm()