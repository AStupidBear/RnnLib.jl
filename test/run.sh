export CUDA_VISIBLE_DEVICES=-1
python src/keras2tf.py --input_model=rnn.h5 --output_model=rnn.pb --output_nodes_prefix=output_
# python -m tensorflow.python.tools.freeze_graph --input_saved_model_dir=rnn --output_node_names='conv1d_5'
python -m tensorflow.python.tools.optimize_for_inference --input=rnn.pb --output=rnn_opt.pb --input_names=input_1 --output_names=output_0