# Usage

We use singularity to run our training on the remote system.
Build `singularity build tensorflow_gpu.simg tensorflow_gpu.recipe`.
To do this through pycharm we point the python interpreter to `$HOME/git/neural_deprojection/singularity/spython_tf.sh`.
Note, this will cause your pycharm to think you don't have python installed. 
You'll see red everywhere in your project.
This is because `spython_tf.sh` is a script that passes your arguments to a singularity image that contains python.

# Temporary solution

For now, when you are editting your code, change your interpretter to a local (proper) interpreter.
This will let code completion work.
Then you'll need to swap the interpreter back to the `spython_tf.sh` when you want to run remotely.


