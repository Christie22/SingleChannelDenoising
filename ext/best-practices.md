### external libraries
- include only if necessary (e.g. the needed feature is a complex algorithm)
- strive to install it using conda
- if not possible, follow conda's guidelines on how to install external packages within the environment
- regenerate dependency files to allow collaborators to stay in track (check `README.md`)

### code structure
- split complex functions in sub-tasks (e.g. metrics: one function for each metric rather than a large one returning a dict)
- if several functionalities are deeply interconnected, write classes instead of functions to promote code reuse and keep a consistent interface
- document the code by specifying input arguments and return values
- in `libs/`, only add or reference code that is needed for the actual script processing (training, denoising, results, etc)
- if the code is used reparately from the main execution (e.g. generate rirs, experiments, tests etc), it belongs to `tools/` or `notebooks/`
- if a file is becoming too disorganized or complex (e.g. lots of loosely connected functions), consider splitting it into logical units; evaluate the advantages and disadvantages of doing so before proceeding

### integrity
- test script on remote server to make sure than no disruptive change has been introduced
- only push to `origin/master` if sure that everything works for everyone