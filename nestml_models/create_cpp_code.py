from pynestml.frontend.pynestml_frontend import to_nest, install_nest

to_nest(input_path="/Users/Jette/GitRepo/clock_network/nestml_models/iaf_cond_diff_exp.nestml",
        target_path="cpp_code",
        logging_level="ERROR")

install_nest("cpp_code", "/Users/Jette/GitRepo/nest-simulator/install")