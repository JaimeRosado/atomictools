# Solucion del modelo Thomas Fermi para átomos o iones positivos
# F. Sanchez Ferrero 1991, F.Blanco 1992-2000-2005
# F.Blanco 2003 añade ayuda para estimar x0 interpolando
# F.Blanco 2005 incluye caso ats. neutros
# F.Blanco 2016 versión C
# J. Rosado 2017 corregidos problemas con decimales y cálculo de x0 interpolado
# F.Blanco 2021 texto en inglés
# F.Blanco 2022 versión python
# J.Rosado 2022 sustituye ventana tkinter por gráfico interactivo de plotly

import numpy as np
from math import sqrt, exp
from ipywidgets import interactive, fixed
import ipywidgets as widgets
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output

#### Cálculo de funciones ####
##############################
def _calc_Fi(Z, N, npt, x0): # resuelve numéricamente Fi()    
    # Inicia x, xs y Fi
    H = x0 / npt
    H12 = H * H / 12.
    x = np.linspace(0., x0, npt+1)
    sx = np.sqrt(x)
    Fi = np.zeros_like(x)
 
    if N==Z: # caso átomo neutro, Fi analítica
        ifi = 0.006944
        ifi = ifi * sx + 0.007298
        ifi = ifi * sx + 0.2302
        ifi = ifi * sx - 0.1486
        ifi = ifi * sx + 1.243
        ifi = ifi * sx + 0.02747
        ifi = ifi * sx + 1.
        Fi = 1. / ifi

    else: # caso ión, cálculo numérico
        Fi[npt]=0.
        Fi[npt-1] = (float(Z-N)/Z) * H / x0
        for i in range(npt-2, 0, -1):
            A = 2. * (1. + 5. * H12 * sqrt(Fi[i+1] / x[i+1]))
            B =       1. -      H12 * sqrt(Fi[i+2] / x[i+2])
            Fi[i] = 2. * Fi[i+1] - Fi[i+2] # estimación de partida
            while True:         # itera la estimación de Fi[i-1] para numerov
                Fest = Fi[i]
                C= 1. - H12 * sqrt(Fest / x[i])
                Fi[i] = (A * Fi[i+1] - B * Fi[i+2]) / C
                if ((Fi[i] / Fest - 1)<1.e-6): break  # hasta autoconsistencia
        Fi[0] = 2. * Fi[1] + H * H * Fi[1] * sqrt(Fi[1] / x[1]) - Fi[2] # extrapola Fi[0]

    # Ya tenemos la Fi, evaluo integral carga como chequeo
    IntRho = (Fi * np.sqrt(Fi) * sx).sum()
    IntRho += sqrt(H) * (19. + 9. * Fi[1]) / 60. # correccion por integral en el primer dx
    IntRho *= Z * H

    return x, Fi, IntRho

def _calc_Rho_Zef(Z, N, x0, x, Fi): # calcula rho y Zef
    K = .88534 / Z**(1/3)
    r0 = K * x0
    r = K * x[1:]
    Ef = -(Z - N) / r0  # r=Kx, EFermi
    tmp = Z / (4. * 3.141593 * K**3)
    Rho = tmp * (Fi[1:] / x[1:])**1.5
    Zef = Z * Fi[1:] - Ef * r # Zeff=-r*V(r)

    return r0, K, Ef, r, Rho, Zef


#### Función para guardar a fichero ####
########################################
def save(Z, N, npt, x0, x, Fi, r0, K, Ef, r, Rho, Zef, filename):
    f = open(filename, "w")
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S       '))
    f.write("Solution of the Thomas Fermi model for atoms" + "\n")

    f.write("Z" + "\t" + "N" + "\t" + "N.Points" + "\t" + "x0" + "\t" + " r0" + "\t" + " K" + "\t" + "Ef" + "\n")
    f.write(str(Z) + "\t" + str(N) + "\t" + str(npt) + "\t\t")
    f.write(f"{x0:<6.4}\t{K*x0:< 6.4}\t{K:< 6.4}\t{Ef:< 6.4}\n")

    f.write(" i     x           Fi          r           rho          V\n")
    f.write(f"{0: 4}  {x[0]:< 5.3e}  {Fi[0]:< 5.3e}  {K*x[0]:< 5.3e}   inf         -inf\n")
    for i in range(1, npt+1):
        f.write(f"{i: 4}  {x[i]:< 5.3e}  {Fi[i]:< 5.3e}  {K*x[i]:< 5.3e}  {Rho[i-1]:< 5.3e}   {-Zef[i-1]/(K*x[i]):< 5.3e}\n")
    f.close


#### Creación de figuras ####
#############################
# Valores por defecto
Z = 50
N = 45
npt = 500
x0 = 10.
x, Fi, IntRho = _calc_Fi(Z, N, npt, x0)
r0, K, Ef, r, Rho, Zef = _calc_Rho_Zef(Z, N, x0, x, Fi)

# Prepara la figura de Fi
fig_Fi = go.Figure(
    data=[go.Scatter(x=x, y=Fi, mode="lines+markers")],
    layout=go.Layout(
        height=400,
        width=900,
        xaxis_title_text='x',
        yaxis_title_text='Fi(x)',
        xaxis_zerolinecolor="Green",
        yaxis_zerolinecolor="Green",
        yaxis_range=[-0.1, 1.3],
    )
)
fig_Fi.update_layout(title_text='Resulting Fi function     ' + f'Fi(0)={Fi[0]:2.3}    Integrated charge={IntRho:3.5}')
# Traza modificable de la figura
line_Fi, = fig_Fi.data

# Prepara figura de Rho y Zef
fig_Rho_Zef = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Charge density", "Effective charge"]
)
fig_Rho_Zef.update_layout(title_text=f'Z={Z:3}    N={N:3}    npt={npt:4}    x0={x0:3.5}    r0={r0:3.5}')
# Plot rho a la izquierda
fig_Rho_Zef.update_xaxes(title_text="r", range=[0., K*x0], row=1, col=1)
fig_Rho_Zef.update_yaxes(title_text="log(rho)", type="log", row=1, col=1)
fig_Rho_Zef.add_trace(go.Scatter(x=r, y=Rho, mode="lines+markers"), row=1, col=1)
# Plot Zef a la derecha
fig_Rho_Zef.update_xaxes(title_text="r", range=[0., K*x0], row=1, col=2)
fig_Rho_Zef.update_yaxes(title_text="Zeff = -r*V(r)", range=[0., Z], row=1, col=2)
fig_Rho_Zef.add_trace(go.Scatter(x=r, y=Zef, mode="lines+markers"), row=1, col=2)
# Objetos modificables de la figura 2
line_Rho, line_Zef = fig_Rho_Zef.data

# Genera/actualiza fig_Fi
def calculate_Fi(Z=Z, N=N, npt=npt, x0=x0):
    x, Fi, IntRho = _calc_Fi(Z, N, npt, x0)

    fig_Fi.update_layout(title_text='Resulting Fi function     ' + f'Fi(0)={Fi[0]:2.3}    Integrated charge={IntRho:3.5}')
    line_Fi.x = x
    line_Fi.y = Fi
    fig_Fi.show()

    return Z, N, npt, x0, x, Fi

# Genera/actualiza fig_Rho_Zef
def calculate_Rho_Zef(Z=Z, N=N, npt=npt, x0=x0, x=x, Fi=Fi):
    r0, K, Ef, r, Rho, Zef = _calc_Rho_Zef(Z, N, x0, x, Fi)

    fig_Rho_Zef.update_layout(title_text=f'Z={Z:3}    N={N:3}    npt={npt:4}    x0={x0:3.5}    r0={r0:3.5}')

    #Plot rho a la izquierda
    line_Rho.x = r
    line_Rho.y = Rho
    fig_Rho_Zef.update_xaxes(range=[0., K*x0], row=1, col=1)

    #Plot Zef a la derecha
    line_Zef.x = r
    line_Zef.y = Zef
    fig_Rho_Zef.update_xaxes(range=[0., K*x0], row=1, col=2)
    fig_Rho_Zef.update_yaxes(range=[0., Z], row=1, col=2)

    fig_Rho_Zef.show()

    return r0, K, Ef, r, Rho, Zef


#### Widgets de las figuras interactivas ####
#############################################
Z_text = widgets.BoundedIntText(
    value=Z,
    min=0,
    max=100,
    step=1,
    description='Nuclear Z:',
    disabled=False
)

N_text = widgets.BoundedIntText(
    value=N,
    min=0,
    max=50,
    step=1,
    description='N electrons:',
    disabled=False
)

npt_text = widgets.BoundedIntText(
    value=npt,
    min=10,
    max=5000,
    step=1,
    description='N. of points:',
    disabled=False
)

x0_text = widgets.BoundedFloatText(
    value=x0,
    min=0.,
    max=20.,
    step=0.01,
    description='Estimated x0:',
    disabled=False
)

accept_button = widgets.Button(
    description="Accept solution"
)

# Caja con el nombre del fichero y botón guardar
file_text = widgets.Text(
    value='ThFermi.txt',
    placeholder='Input file name',
    description='File name:',
    disabled=False
)
save_button = widgets.Button(description="Save")
save_box = widgets.Box([file_text, save_button])

# Genera un link entre widgets para asegurar N<=Z
dl = widgets.dlink((Z_text, 'value'), (N_text, 'max'))

# Activa/desactiva x0_text para átomos ionizados/neutros
def update_x0(Z, N):
    if N==Z: # caso átomo neutro
        # Desactiva el widget de x0
        x0_text.disabled = True
    else: # caso ion
        # Activa el widget de x0
        x0_text.disabled = False


#### Función para crear figuras interactivas ####
#################################################
def calculate():
    # Genera gráfico interactivo y botón de aceptar
    output_Fi = interactive(calculate_Fi, Z=Z_text, N=N_text, npt=npt_text, x0=x0_text);
    x0_control = interactive(update_x0, Z=Z_text, N=N_text);
    display(output_Fi)
    display(accept_button)
    
    def Accept(b): # Solucion aceptada, mostrar resultados
        clear_output();
    
        #   r0, K, Ef, r, Rho, Zef = calculate_Rho_Zef(*output.result)
        output_Rho_Zef = calculate_Rho_Zef(*output_Fi.result)
        display(save_box)

        def Save(b): # Grabar en fichero
            clear_output();
            #save(Z, N, npt, x0, x, Fi, r0, K, Ef, r, Rho, Zef, file_text.value)
            save(*output_Fi.result, *output_Rho_Zef, file_text.value)

        save_button.on_click(Save)

    accept_button.on_click(Accept)


#### Función para comprobación interactiva del ajuste del potencial ####
########################################################################
from orbitals import _pot_GSZ, H_text, D_text

# Valores por defecto
V = np.zeros_like(r)
V = -Zef/r

fig_V = go.Figure(
    data=[
        go.Scatter(x=r, y=V),
        go.Scatter(x=r, y=V)
    ],
    layout=go.Layout(
        height=400,
        width=900,
        xaxis_title_text='r',
        yaxis_title_text='V(r)',
        xaxis_zerolinecolor="Green",
        yaxis_zerolinecolor="Green",
    )
)
line_V, line_V_GSZ = fig_V.data

def update_fig_V(r, V, Z, N, H, D):
    V_GSZ = _pot_GSZ(r, Z, N, H, D)
    rmin = r[r<0.1].max()
    rmc = r[-1]
    Vmin = V[r==rmin][0]

    line_V.x = r[1:]
    line_V.y = V[1:]
    line_V_GSZ.x = r[1:]
    line_V_GSZ.y = V_GSZ[1:]
    fig_V.update_xaxes(range=[-0.01*rmc, 1.01*rmc])
    fig_V.update_yaxes(range=[Vmin, -0.01*Vmin])    
    fig_V.show()

# Botón para cargar fichero
load_button = widgets.Button(description="Load")
load_box = widgets.Box([file_text, load_button])

def check_fit():
    display(load_box)

    def Load(b): # Cargar fichero
        clear_output();
        filename = file_text.value
        Z, N = np.loadtxt(filename, skiprows=2, unpack=True, usecols = (0, 1), max_rows=1)
        r, V = np.loadtxt(filename, skiprows=5, unpack=True, usecols = (3, 5))

        # Activa H_text y D_text
        # (pueden estar desactivados si se está ejecutando orbitals con otro pontecial)
        H_text.disabled = False
        D_text.disabled = False
        # Genera figura interactiva donde se pueden cambiar H y D
        output_V = interactive(update_fig_V, r=fixed(r), V=fixed(V), Z=fixed(Z), N=fixed(N),
                               H=H_text, D=D_text)
        display(output_V)

    load_button.on_click(Load)
    
    
    
