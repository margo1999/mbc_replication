""" wo ist Nutzer installation von NEST
"""
# TODO documentation

if __name__ == '__main__':

    import os
    import pynestml.frontend.pynestml_frontend

    CLOCK_NETWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(f"{CLOCK_NETWORK_ROOT=}")

    NESTML_INPUT = os.path.join(CLOCK_NETWORK_ROOT, "nestml_models", "iaf_cond_diff_exp.nestml")
    NESTML_OUTPUT = os.path.join(CLOCK_NETWORK_ROOT, "nestml_models", "generated_code")
    NEST_SIMULATOR_INSTALL_LOCATION = os.path.join(os.path.dirname(CLOCK_NETWORK_ROOT), "nest-simulator", "install")
    print(f"{NEST_SIMULATOR_INSTALL_LOCATION=}")

    if hasattr(pynestml.frontend.pynestml_frontend, "generate_nest_target"):
        from pynestml.frontend.pynestml_frontend import generate_nest_target  # type: ignore # pylint: disable=no-name-in-module

        generate_nest_target(input_path=NESTML_INPUT,
                             target_path=NESTML_OUTPUT,
                             logging_level="ERROR",
                             codegen_opts={"nest_path": NEST_SIMULATOR_INSTALL_LOCATION})

    elif hasattr(pynestml.frontend.pynestml_frontend, "to_nest") and hasattr(pynestml.frontend.pynestml_frontend, "install_nest"):
        from pynestml.frontend.pynestml_frontend import to_nest, install_nest

        to_nest(input_path=NESTML_INPUT, target_path=NESTML_OUTPUT, logging_level="ERROR")

        install_nest(NESTML_OUTPUT, NEST_SIMULATOR_INSTALL_LOCATION)
    else:
        raise Exception("Cannot handle NESTML version")
