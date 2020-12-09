from functools import partial

from jax import lax, make_jaxpr
from jax.experimental import jax2tf

import numpy as np
import tensorflow as tf

# NOTE: do not use this. It depends on internal APIs that are subject to
# change.

def tf2jax(tf_graph):
  def _decode_jax_op(s):
    # prim name to prim
    jax_ops = { "dot_general_p": lax.dot_general_p
              , "add_p": lax.add_p
              }
    prim_call = bytes.fromhex(s).decode('latin1').split(' ')
    prim = jax_ops[prim_call[0]]

    kwargs = {}
    for kwarg in prim_call[1:]:
      kw, val = kwarg.split('=')
      # let's not do that for real
      kwargs[kw] = eval(val)

    return partial(prim.bind, **kwargs)
    
  def _make_partial_function(fun, inputs, funcs, debug='lol'):
    return (lambda func, args, func_dict: lambda *delayed_args: lambda: fun(*[func_dict[a](*delayed_args)() for a in args]))(fun, inputs, funcs)
  partial_functions = {}
  for node in tf_graph.as_graph_def().node:
    if node.name.startswith('args_'):
      #args.add(node.name)
      partial_functions[node.name] = (lambda node_name: lambda *delayed_args: lambda: delayed_args[int(node_name[5:])])(node.name)
    elif len(node.input) == 0:
      # ignore consts in poc
      continue
    elif node.name.startswith('JAX_'):
      cmd = node.name.split('/')[0].split('_')[1]
      jax_op = _decode_jax_op(cmd)
      partial_functions[node.name] = _make_partial_function(jax_op, node.input, partial_functions, node.name)
    else:
      if not node.name.startswith('Identity'):
        # inserted "metadata" ops by TF?
        continue
      # identity functions
      identity = lambda *args: args[0]
      partial_functions[node.name] = _make_partial_function(identity, node.input, partial_functions, node.name)
  return (lambda *args: partial_functions[tf_graph.as_graph_def().node[-1].name](*args)())

def f_jax(arr1, arr2, arr3):
  return lax.dot(lax.add(lax.dot(arr1, arr2), arr3), arr3)

arrs = [ np.ones((4, 5), dtype=np.float32)
       , np.ones((5, 4), dtype=np.float32)
       , np.ones((4, 4), dtype=np.float32)
       ]

print(make_jaxpr(f_jax)(*arrs))

f_tf = (tf.function(jax2tf.convert(f_jax), autograph=False)
          .get_concrete_function(*list(map(tf.convert_to_tensor, arrs))))

tf_graph = f_tf.graph
f_jax2 = tf2jax(tf_graph)

print(make_jaxpr(f_jax2)(*arrs))
