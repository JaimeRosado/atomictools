# Solucion eq. Schrodinger para potencial central por Numerov paso variable
# J.Rosado 2022 sustituye ventana tkinter por gráfico interactivo de plotly

import numpy as np
from math import sqrt, exp
from ipywidgets import interactive, fixed, HBox, VBox
import ipywidgets as widgets
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#### Cálculo de funciones ####
##############################
def _pot_Hdg(r, Z): # calcula potencial hidrogenoide
    V = np.zeros_like(r)
    V[0] = -np.inf
    V[1:] = -Z / r[1:] # V[0]=0, V[1]=...
    return V

def _pot_SdR(r, Z, A, a1, a2): # calcula potencial de Sánchez del Río
    V = np.zeros_like(r)
    V[0] = -np.inf
    V[1:] = -((Z - .5) * (A * np.exp(-a1*r[1:]) + (1.-A) * np.exp(-a2*r[1:])) + 1.) / r[1:]
    return V

def _pot_GSZ(r, Z, N, H, D): # calcula potencial de Green-Sellin-Zachor
    V = np.zeros_like(r)
    V[0] = -np.inf
    V[1:] = -(Z - N + 1 + (N-1) / (1. + H * (np.exp(r[1:]/D) - 1.))) / r[1:]
    V[r/D>75.] = -(Z - N + 1) / r[r/D>75.]
    return V

def _calc_pot(model, Z, N, A, a1, a2, H, D, npt, rmc):
    cri = rmc / (npt**2)  # factor conversion r=cri*i^2
    r = cri * np.arange(0, npt+1, 1)**2 # r[0]=0, r[1]=cri, r[npt]=rmc
    if model=='Hdg':
        V = _pot_Hdg(r, Z)
    elif model=='SdR':
        V = _pot_SdR(r, Z, A, a1, a2)
    elif model=='GSZ':
        V = _pot_GSZ(r, Z, N, H, D)
    else:
        raise ValueError("Please, input a valid potential model: 'Hdg', 'SdR' or 'GSZ'.")
    return r, V

def _calc(r, V, Z, E, L): # resuelve ecuación de Schrödinger
    "resuelve eq. schrodinger por método Numerov con la f.de o. en p[0 a Npt]"
    npt = len(r) - 1
    f = np.zeros_like(r)
    P = np.zeros_like(r)
    f[1:] = L * (L+1) / (r[1:]**2) - 2. * E + 2. * V[1:]
    # valores de partida de P (los dos últimos)
    P[npt-1:] = 1000. * np.exp(-np.sqrt(-2.*E) * r[npt-1:])
    #min_pr = 0.
    #max_pr = 0.
    if P[npt]==0.:
        raise ValueError("rmc too large. Please, try a smaller value.")

    # r, f y P tienen npt+1 valores
    h = np.diff(r) # npt valores
    h1 = h[:-1]    # npt-1 valores
    h2 = h[1:]     # npt-1 valores
    hh1 = h1 * h1
    hh2 = h2 * h2
    h12 = h1 * h2
    ha = h1 / (h1 + h2)
    hc = h2 / (h1 + h2)
    hD = hc * (hh1 + h12 - hh2) * f[:-2] / 12. - hc
    hN = 1. + (hh1 + 3. * h12 + hh2) * f[1:-1] / 12.
    hM = ha - ha * (-hh1 + h12 + hh2) * f[2:] / 12.
    # iteración Numerov
    for i in range(npt-2, 0, -1): # hasta i=1
        P[i] = (hM[i] * P[i+2] - hN[i] * P[i+1]) / hD[i]

    # pto. mas prox. al origen valorado a efectos de escala y normalización
    cri = r[1]
    i_st = int(1. + np.sqrt(L * (L+1) / (20. * Z * cri)))
    points = range(i_st, npt+1)
    integral = 2. * cri * (points * P[i_st:]**2).sum() # integral |P(r)|^2 dr
    P = P / np.sqrt(integral)

    return P


#### Función para guardar a fichero ####
########################################
def save(model, Z, N, A, a1, a2, H, D, npt, rmc, E, L, r, P, filename):
    f = open(filename,"w")
    f.write( datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" )
    f.write("Calculation of atomic orbitals for a central potential\n")
    f.write("Solving Schrodinger eq. by variable sptep-size Numerov method\n")
    param_text = "L" + "\t" + "E(a.u.)" + "\t  " + "r_max(a.u.)" + "\t" + "N. points"
    param_values = str(L) + "\t" + str(E) + "\t  " + str(rmc) + "\t\t" + str(npt)

    if model=='Hdg':
        f.write("Hydrogenic potential -Z/r" + "\n")
        f.write("Z" + "\t" + param_text + "\n")
        f.write(str(Z) + "\t" + param_values + "\n")
    elif model=='SdR':
        f.write("S. del Rio potential" + "\n")
        f.write("Z" + "\t" + "A" + "\t" + "a1" + "\t" + "a2" + "\t" + param_text + "\n")
        f.write(str(Z) + "\t" + str(round(A, 3)) + "\t" + str(round(a1, 3)) + "\t" + str(round(a2, 3))
                + "\t" + param_values + "\n")
    elif model=='GSZ':
        f.write("Green-Sellin-Zachor potential" + "\n")
        f.write("Z" + "\t" + "N" + "\t" + "H" + "\t" + "D" + "\t" + param_text + "\n")
        f.write(str(Z) + "\t" + str(N) + "\t" + str(round(H, 3)) + "\t" + str(round(D, 3))
                + "\t" + param_values + "\n")

    f.write("Normalised radial wavefunction P(r)" + "\n")
    cri = r[1]
    f.write(f" Format r(i), P(i) with r(i)={cri:8.6}*i^2\n")
    for i in range(1, npt+1):
        f.write(f"{r[i]:< 8.6e}  {P[i]:< 8.6e}\n")
    f.close


#### Creación de figuras ####
#############################
# Valores por defecto
model = 'Hdg'
Z = 1
N = 1
A = 0.
a1 = 0.
a2 = 0.
H = 0.
D = 1.
npt = 300
rmc = 10.
E = -0.5
L = 0
r, V = _calc_pot(model, Z, N, A, a1, a2, H, D, npt, rmc)
P = _calc(r, V, Z, E, L)


#### Widgets de las figuras interactivas ####
#############################################
# Define los widgets para interaccionar con la figura
model_dropdown = widgets.Dropdown(
    options=[('Hydrogenic', 'Hdg'), ('Sánchez del Río', 'SdR'), ('Green-Sellin-Zachor', 'GSZ')],
    value='Hdg',
    description='P. model:',
    disabled=False,
)

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
    max=100,
    step=1,
    description='N electrons:',
    disabled=False
)

A_text = widgets.BoundedFloatText(
    value=A,
    min=0.,
    max=2.,
    step=0.001,
    description='A:',
    disabled=False
)

a1_text = widgets.BoundedFloatText(
    value=a1,
    min=0.,
    max=3.,
    step=0.001,
    description='a1:',
    disabled=False
)

a2_text = widgets.BoundedFloatText(
    value=a2,
    min=0.,
    max=20.,
    step=0.001,
    description='a2:',
    disabled=False
)

H_text = widgets.BoundedFloatText(
    value=H,
    min=0.,
    max=10.,
    step=0.001,
    description='H:',
    disabled=False
)

D_text = widgets.BoundedFloatText(
    value=D,
    min=0.,
    max=2.,
    step=0.0001,
    description='D:',
    disabled=False
)

npt_text = widgets.BoundedIntText(
    value=npt,
    min=10,
    max=1000,
    step=1,
    description='N. of points:',
    disabled=False
)

rmc_text = widgets.BoundedFloatText(
    value=rmc,
    min=0.1,
    max=200.,
    step=0.1,
    description='Max. radius:',
    disabled=False
)

E_text = widgets.BoundedFloatText(
    value=E,
    min=-2000.,
    max=0.,
    step=0.01,
    description='E (a.u.):',
    disabled=False
)

L_text = widgets.BoundedIntText(
    value=L,
    min=0,
    max=10,
    step=1,
    description='L:',
    disabled=False
)

file_box = widgets.Text(
    value='orbital.txt',
    placeholder='Input file name',
    description='File name:',
    disabled=False
)
save_button = widgets.Button(description="Save")
save_box = widgets.Box([file_box, save_button])

#left = VBox([model_dropdown, Z_text, N_text, A_text, a1_text, a2_text, H_text, D_text])
#right = VBox([npt_text, rmc_text, E_text, L_text, file_box])
#control = HBox([left, right])

# Genera un link entre widgets para asegurar N<=Z
dl = widgets.dlink((Z_text, 'value'), (N_text, 'max'))

# Activa/desactiva parámetros según modelo de potencial
# y activa botón Save en cada modificación de parámetros
def update_widgets(model, Z, N, A, a1, a2, H, D, npt, rmc, E, L):
    if model=='Hdg':
        # Desactiva widgets de N, A, a1, a2, H, D
        N_text.disabled = True
        A_text.disabled = True
        a1_text.disabled = True
        a2_text.disabled = True
        H_text.disabled = True
        D_text.disabled = True

    elif model=='SdR':
        # Activa widgets de A, a1, a2
        A_text.disabled = False
        a1_text.disabled = False
        a2_text.disabled = False
        # Desactiva widgets de N, H, D
        N_text.disabled = True
        H_text.disabled = True
        D_text.disabled = True
        # Da el valor de N=Z porque SdR asume átomo neutro
        N_text.value = Z

    elif model=='GSZ':
        # Activa widgets de N, H, D
        N_text.disabled = False
        H_text.disabled = False
        D_text.disabled = False
        # Desactiva widgets de A, a1, a2
        A_text.disabled = True
        a1_text.disabled = True
        a2_text.disabled = True

    # Activa botón de guardar y pone texto "Save"
    # en caso de que no esté hecho
    save_button.disabled = False
    save_button.description = "Save"


#### Función para crear figura interactiva ####
###############################################
def calculate():
    # Prepara la figura del potencial
    #fig_V, ax_V = plt. subplots(figsize=(9, 4))
    #ax_V.grid(True)
    #ax_V.set_xlabel('r')
    #ax_V.set_ylabel('V(r)')
    #ax_V.axhline(0, color='green', linewidth=0.8)  # Línea horizontal en y=0
    #ax_V.axvline(0, color='green', linewidth=0.8)  # Línea vertical en x=0
    ## Traza modificable de la figura
    #line_V, = ax_V.plot(r[1:], V[1:])

    # Actualiza fig_V
    #def calculate_potential(model=model, Z=Z, N=N, A=A, a1=a1, a2=a2, H=H, D=D, npt=npt, rmc=rmc):
    #    r, V = _calc_pot(model, Z, N, A, a1, a2, H, D, npt, rmc)
    #    rmin = r[r<0.8].max()
    #    Vmin = V[r==rmin][0]
    #
    #    line_V.set_xdata(r[1:])
    #    line_V.set_ydata(V[1:])
    #    ax_V.set_xlim([-0.01*rmc, 1.01*rmc])
    #    ax_V.set_ylim([Vmin, -0.01*Vmin])   
    #    ax_V.relim()  # Recalcular los límites del gráfico basado en los nuevos datos
    #    ax_V.autoscale_view()  # Ajustar la escala de los ejes si es necesario
    #    fig_V.canvas.draw()  # Redibujar la figura
    #
    #    return r, V
    
    # Prepara la figura del orbital
    fig_P, ax_P = plt. subplots(figsize=(9, 4))
    ax_P.grid(True)
    ax_P.set_xlabel('r')
    ax_P.set_ylabel('P(r)')
    ax_P.axhline(0, color='green', linewidth=1.5)  # Línea horizontal en y=0
    ax_P.axvline(0, color='green', linewidth=1.5)  # Línea vertical en x=0
    # Traza modificable de la figura
    line_P, = ax_P.plot(r[1:], V[1:], 'o-')
    line_P2, = ax_P.plot(r[1:0], V[1:0], 'o-')
    # Inset
    axins_P = inset_axes(ax_P, width="40%", height="40%")
    axins_P.axhline(0, color='green', linewidth=1.5)  # Línea horizontal en y=0
    axins_P.axvline(0, color='green', linewidth=1.5)  # Línea vertical en x=0
    # Traza modificable del inset
    insline_P, = axins_P.plot(r[1:10], V[1:10], 'o-')
    insline_P2, = axins_P.plot(r[1:0], V[1:0], 'o-')
    
    def calculate_P(model=model, Z=Z, N=N, A=A, a1=a1, a2=a2, H=H, D=D, npt=npt, rmc=rmc, E=E, L=L):
        r, V = _calc_pot(model, Z, N, A, a1, a2, H, D, npt, rmc) # No se usa calculate_potential
        P = _calc(r, V, Z, E, L)
        cri = rmc/(npt**2);
        i_st=int(1+sqrt(L*(L+1)/(20*Z*cri)))
        ins_x=max(10, i_st+5)
        line_P.set_xdata(r[1:])
        line_P.set_ydata(P[1:])
        line_P2.set_xdata(r[1:i_st])
        line_P2.set_ydata(P[1:i_st])
        ax_P.set_xlim([-0.01*rmc, 1.01*rmc])
        insline_P.set_xdata(r[1:ins_x])
        insline_P.set_ydata(P[1:ins_x])
        insline_P2.set_xdata(r[1:i_st])
        insline_P2.set_ydata(P[1:i_st])
        rmax = r[P==P.max()]
        rmax = rmax[0]
        fig_P.suptitle('Maximum of P at ' + f'{rmax:2.3}')
        ax_P.relim()  # Recalcular los límites del gráfico basado en los nuevos datos
        ax_P.autoscale_view()  # Ajustar la escala de los ejes si es necesario
        axins_P.relim()  # Recalcular los límites del gráfico basado en los nuevos datos
        axins_P.autoscale_view()  # Ajustar la escala de los ejes si es necesario
        fig_P.canvas.draw()  # Redibujar la figura

        return model, Z, N, A, a1, a2, H, D, npt, rmc, E, L, r, P

    # Genera gráfico interactivo del orbital
    # Nótese que calculate_P incluye _calc_pot
    output = interactive(calculate_P, model=model_dropdown, Z=Z_text, N=N_text,
                         A=A_text, a1=a1_text, a2=a2_text,
                         H=H_text, D=D_text,
                         npt=npt_text, rmc=rmc_text,
                         E=E_text, L=L_text);
    widget_control = interactive(update_widgets, model=model_dropdown, Z=Z_text, N=N_text,
                         A=A_text, a1=a1_text, a2=a2_text,
                         H=H_text, D=D_text,
                         npt=npt_text, rmc=rmc_text,
                         E=E_text, L=L_text);
    display(output)
    display(save_box)

    def Save(b): #grabar en fichero
        # Desactiva el botón y cambia texto
        save_button.disabled = True
        save_button.description = "Done"

        # Guarda el fichero
        filename = file_box.value
        #save(model, Z, N, A, a1, a2, H, D, npt, rmc, E, L, r, P, filename)
        save(*output.result, filename)

    save_button.on_click(Save)