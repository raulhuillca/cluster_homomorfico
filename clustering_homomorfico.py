import numpy as np

#Definimos datos de prueba
np.random.seed(1000)

N = 25 * 25

x0,x1 = np.mgrid[5.0:5.0 + 0.5 * 25:0.5, 5.0:5.0 + 0.5 * 25:0.5]
X = np.vstack((x0.flatten(), x1.flatten())).T

y = x0.flatten() * 5.1 + x1.flatten() * 2.1 + 1.5 + np.random.normal(0.0, 2.0, size=N)


#Entrenamos el modelo
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y)

# Definioms el programa
from eva import *
poly = EvaProgram("Polynomial", vec_size=1024)
with poly:
    x0 = Input("x0")
    x1 = Input("x1")
    Output("y", reg.coef_[0]*x0 + reg.coef_[1]*x1 + reg.intercept_)
poly.set_output_ranges(30)
poly.set_input_scales(30)
# Compilamos el programa con el esquema CKKS
from eva.ckks import *
compiler = CKKSCompiler()
compiled_poly, params, signature = compiler.compile(poly)

from eva import save, load
save(params, 'poly.evaparams')


# Generamos claves
from eva.seal import *
params = load('poly.evaparams')
public_ctx, secret_ctx = generate_keys(params)
# Guardamos la clave publica
save(public_ctx, 'poly.sealpublic')

# Creamos el cifrado para x0 y x1
inputs = { "x0": [0.0 for i in range(compiled_poly.vec_size)], "x1": [0.0 for i in range(compiled_poly.vec_size)] }
inputs["x0"][0] = 1.1
inputs["x0"][1] = 2.2
inputs["x0"][2] = 3.3
inputs["x1"][0] = 1.1
inputs["x1"][1] = 2.2
inputs["x1"][2] = 3.3
public_ctx = load('poly.sealpublic')
encInputs = public_ctx.encrypt(inputs, signature)
#  Guardar la informacion cifrada de tipo byte en archivo
from eva import save
save(encInputs, 'poly_inputs.sealvals')

# Procesamos con homomorphic encryption (HE)
public_ctx = load('poly.sealpublic')
encOutputs = public_ctx.execute(compiled_poly, encInputs)
# Resultados
save(encOutputs, 'poly_outputs.sealvals')

encOutputs = load('poly_outputs.sealvals')
outputs = secret_ctx.decrypt(encOutputs, signature)
print("********** Result is **********")
for i in range(3):
    print(outputs["y"][i])
# Resultados para la ecuacion 5.1*x0 + 2.1*x1 + 0.5
