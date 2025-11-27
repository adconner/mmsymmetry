from sage.all import gap
import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
from itertools import *
import time

from symmetry_search import get_map_to_rank1s, matrixmult, evaluate

numit = 20000
batch = 1000
printevery = 1000
print_num = 5

m,n,l = 4,4,4
cx = True
T = matrixmult(m,n,l)
if cx:
    T = T.astype('complex64')

gap.Read('"compute_symmetry.g"');
gap.execute('''
g := Group([( 1, 4)( 2, 6)( 3, 7)( 5, 8)( 9,12,11,15,10,14,13,17)(16,21,19,23,18,22,20,24),
    ( 1, 2)( 4, 7, 6, 8)( 9,10)(11,13)(12,16)(14,18)(15,19)(17,20),
    ( 1, 3)( 2, 5)( 4, 7, 6, 8)( 9,11,10,13)(12,15,14,17)(16,20,18,19) (21,24,22,23)]);
acts := [
  [ [ [ [ 0, 1, 0, -1 ], [ E(4), 0, -1, 0 ], [ E(4), 0, 1, 0 ],
  [ 0, E(4), 0, E(4) ] ],
    [ [ 0, 0, 0, 1 ], [ 0, 0, -1, 0 ], [ 0, -1, -1, -1 ], [ 1, 0, -1, -1 ] ],
    [ [ 0, 1, 1, 0 ], [ -E(4), 0, 0, 1 ], [ 0, E(4), -E(4), 0 ],
  [ E(4), 0, 0, 1 ] ] ], (2,3)],
  [ [ [ [ 1, 0, 0, 0 ], [ 0, 0, 0, -E(4) ], [ 0, 0, -1, 0 ], [ 0, -E(4), 0, 0 ] ],
    [ [ 0, 1, 1, 0 ], [ -1, 0, 0, 1 ], [ 0, 0, 1, 0 ], [ 0, 0, 0, -1 ] ],
    [ [ 0, 0, 0, 1 ], [ 0, 1, 0, 0 ], [ 0, 0, -1, 0 ], [ -1, 0, 0, 0 ] ] ], ()],
  [ [ [ [ 0, 0, 1, 0 ], [ 0, 1, 0, 0 ], [ -1, 0, 0, 0 ], [ 0, 0, 0, 1 ] ],
    [ [ 0, 1, 1, 1 ], [ -1, 0, 0, 0 ], [ 0, 0, 0, 1 ], [ 0, 0, 1, 0 ] ],
    [ [ 0, 0, 0, 1 ], [ 0, E(4), 0, 0 ], [ 0, 0, -E(4), 0 ], [ 1, 0, 0, 0 ] ] ], ()] ];
fac_perm_map := GroupHomomorphismByImages(g, Group(List(acts, p->p[2])));
acts := List(acts, p -> TripToMat(p[1], p[2]));
hom := GroupHomomorphismByImages(g,Group(acts));
tripf := e -> MatToTrip(e ^ hom, [4,4,4])[1];
orbits := [
  [ Group(
[ ( 1, 4)( 2, 6)( 3, 7)( 5, 8)( 9,12,11,15,10,14,13,17)(16,21,19,23,18,22,20,
     24), ( 1, 6)( 2, 4)( 3, 8)( 5, 7)( 9,12,11,15,10,14,13,17)
    (16,21,19,23,18,22,20,24), ( 1, 2)( 3, 5)( 4, 6)( 7, 8)(12,14)(15,17)
    (16,18)(19,20) ] ),
  [ [ [ -E(4), 0, 0 ], [ 0, 0, 1+E(4) ], [ 0, 1/2+1/2*E(4), 0 ] ],
    [ [ -E(4), 0, 0 ], [ 0, 0, -1-E(4) ], [ 0, -1/2-1/2*E(4), 0 ] ],
    [ [ -1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, -1 ] ] ] ],
  [ Group(
[ ( 1, 4, 2, 6)( 3, 7, 5, 8)( 9,17,13,14,10,15,11,12)(16,24,20,22,18,23,19,21)
    , ( 1, 7)( 2, 8)( 3, 4)( 5, 6)( 9,18,11,20,10,16,13,19)(12,21,15,23,14,22,
     17,24), (12,14)(15,17)(16,18)(19,20), (1,2)(3,5)(4,6)(7,8) ] ),
  [ [ [ -E(4), 0, 0 ], [ 0, 0, 1 ], [ 0, E(4), 0 ] ],
    [ [ -E(4), 0, 0 ], [ 0, 0, -1 ], [ 0, -E(4), 0 ] ],
    [ [ -1, 0, 0 ], [ 0, -1, 0 ], [ 0, 0, 1 ] ],
    [ [ 1, 0, 0 ], [ 0, -1, 0 ], [ 0, 0, -1 ] ] ] ] ];
''')
            
rep = {'g' : gap('g'), 'tripf' : gap('tripf'), 'fac_perm_map' : gap('fac_perm_map')}
orbits = gap('orbits')
rank1s, params, embeds = get_map_to_rank1s(rep,orbits)

def init(key):
    dshape = jnp.zeros((params,),jnp.complex64 if cx else jnp.float32)
    return optax.tree.random_like(key, dshape, jax.random.normal)

key = jax.random.key(np.random.randint(2**63))
key, keycur = jax.random.split(key)
x = jax.vmap(init)(jax.random.split(keycur,batch))

@jax.jit
def loss_fn(x,it,key):
    dec = rank1s(x,jnp)
    S = jnp.einsum('ir,jr,kr->ijk', *dec)
    E = S-T
    reconstruction_loss = jnp.mean(jnp.real(E*E.conj()))

    dec_quantized = jax.tree.map(lambda f: jnp.round(f*2)/2,dec)
    S_quantized = jnp.einsum('ir,jr,kr->ijk', *dec_quantized)
    E_quantized = S_quantized - T
    quantized_reconstruction_loss = jnp.mean(jnp.real(E_quantized*E_quantized.conj()))
    quantization_loss = 0.0
    for f in dec:
        quantization_loss += 0.005 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))

    loss = reconstruction_loss + quantization_loss
  
    return loss, { 'reconstruction loss' : reconstruction_loss, 'quantization loss': quantization_loss,
                  'quantized_reconstruction_loss' : quantized_reconstruction_loss }
  
start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(x)

@functools.partial(jax.jit,donate_argnums=[0,1])
@functools.partial(jax.vmap, in_axes=[0,0,0,None])
def update_function(x, opt_state, key, it):
    (loss, info), grads = jax.value_and_grad(loss_fn,has_aux=True)(x,it,key)
    grads = jax.tree.map(jnp.conj, grads)
    updates, opt_state = optimizer.update(grads, opt_state, x, value = loss, grad = grads)
    x = optax.apply_updates(x, updates)
    return x, opt_state, loss, info

# Some user feedback for interactive use
@jax.jit
def extra_info(opt_state, loss, x, info):
    lossmult = T.size
    solutions = jnp.count_nonzero(info['quantized_reconstruction_loss'] == 0.0)
    besti = jnp.argpartition(loss,print_num-1)[:print_num]
    besti = besti[jnp.argsort(loss[besti])]
    x = jax.tree.map(lambda e: e[besti], x)
    info = jax.tree.map(lambda e: e[besti]*lossmult, info)
    info['example index'] = besti
    maxabs = jax.vmap(lambda x: jnp.max(jnp.abs(x)))(x)
    info['maximum coefficient'] = maxabs
    info['solutions'] = solutions
    return info

for it in range(numit):
    if it == 1:
        startt = time.time()
    key, keycur = jax.random.split(key)
    x, opt_state, loss, info = update_function(x, opt_state, jax.random.split(keycur,batch), it)
    if it % printevery == printevery-1 or it == numit-1:
        info = extra_info(opt_state,loss,x,info)
        elapsed = time.time() - startt
        print(f'{it}: {elapsed:.3f}s elapsed, {it*batch / (1000*elapsed):.0f}K iteratons/s')
        for desc, v in sorted(info.items()):
            print(desc.rjust(25), v)
      
