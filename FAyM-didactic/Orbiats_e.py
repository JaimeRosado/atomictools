# Solving Schrodinger eq. for central potentials by variable step size Numerov method
# Francisco Blanco: c version in 2005
# Francisco Blanco: python english version in 2021
# Jaime Rosado: output format adapted to atomictools

import tkinter #importa módulo de ventanas y botones
from math import sqrt,exp

def runHdg():       #activa potencial Hidrogenoide
    global TipoV; TipoV="Hdg"; disables()
    Ent_Hdg_Z["state"]="normal"

def runSdR():       #activa potencial S del Rio
    global TipoV; TipoV="SdR"; disables()
    Ent_SdR_Z["state"]="normal";  Ent_SdR_A["state"]="normal"
    Ent_SdR_a1["state"]="normal"; Ent_SdR_a2["state"]="normal"

def runGSZ():       #activa potencial Green-Sell-Zach
    global TipoV; TipoV="GSZ"; disables()
    Ent_GSZ_Z["state"]="normal";  Ent_GSZ_H["state"]="normal"
    Ent_GSZ_N["state"]="normal";  Ent_GSZ_D["state"]="normal"

def Fpotencial(r): #devuelve funcion f con el potencial elegido
    L=int(Ent_C_L.get()); E=float(Ent_C_E.get()); Z=float(strZ.get())
    if TipoV=="Hdg":
        V=-Z/r
    elif TipoV=="SdR":
        A=float(Ent_SdR_A.get())
        a1=float(Ent_SdR_a1.get()); a2=float(Ent_SdR_a2.get())
        V=-((Z-.5)*(A*exp(-a1*r)+(1-A)*exp(-a2*r))+1)/r
    elif TipoV=="GSZ":
        N=float(Ent_GSZ_N.get())
        H=float(Ent_GSZ_H.get());   D=float(Ent_GSZ_D.get())
        if (r/D>75): V=-(Z-N+1)/r 
        else: V=-(Z-N+1+(N-1)/(1+H*(exp(r/D)-1)))/r
    return L*(L+1)/(r**2)-2*E+2*V

from datetime import datetime #manipulación de fechas y horas
def runSave():
    runCalc() #primero actualiza calculo por si ha habido cambios
    Npt=int(Ent_C_Npt.get()); Rmc=float(Ent_C_Rmc.get()); cri=Rmc/(Npt**2)
    f=open(Ent_F.get(),"w")
    f.write( datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n" )
    f.write("Calculation of atomic orbitals for a central potential"+"\n")
    f.write("Solving Schrodinger eq. by variable sptep-size Numerov method"+"\n")
    param_text="L"+"\t"+"E(a.u.)"+"\t  "+"r_max(a.u.)"+"\t"+"N. points"
    param_values=Ent_C_L.get()+"\t"+Ent_C_E.get()+"\t  "+Ent_C_Rmc.get()+"\t\t"+Ent_C_Npt.get()
    if TipoV=="Hdg":
        f.write("Hydrogenic potential -Z/r"+"\n")
        f.write("Z"+"\t"+param_text+"\n")
        f.write(Ent_Hdg_Z.get()+"\t"+param_values+"\n")
    if TipoV=="SdR":
        f.write("S. del Rio potential"+"\n")
        f.write("Z"+"\t"+"A"+"\t"+"a1"+"\t"+"a2"+"\t"+param_text+"\n")
        f.write(Ent_SdR_Z.get()+"\t"+Ent_SdR_A.get()+"\t"+Ent_SdR_a1.get()+"\t"+Ent_SdR_a2.get()
                +"\t"+param_values+"\n")
    if TipoV=="GSZ":
       #f.write("Green-Sellin-Zachor potential, Z="+Ent_GSZ_Z.get()+"\n")
       #f.write("N=",Ent_GSZ_N.get()+" H="+Ent_GSZ_H.get()+" D=",Ent_GSZ_D.get()+"/n")
        f.write("Green-Sellin-Zachor potential"+"\n")
        f.write("Z"+"\t"+"N"+"\t"+"H"+"\t"+"D"+"\t"+param_text+"\n")
        f.write(Ent_GSZ_Z.get()+"\t"+Ent_GSZ_N.get()+"\t"+Ent_GSZ_H.get()+"\t"+Ent_GSZ_D.get()
                +"\t"+param_values+"\n")
    f.write("Normalised radial wavefunction P(r) : ")
    f.write("values at "+Ent_C_Npt.get()+" points"+"\n")
    f.write(f" Format r(i),P(i) with r(i)={cri:8.6}*i^2"+"\n")
    for i in range(1,Npt+1):f.write(f"{r[i]:< 8.6e}  {P[i]:< 8.6e}\n")
    f.close

def runCalc():      #si se pulsa actualizar cálculo
    "resuelve eq. schrodinger por método numrov con la f.de o. en p[0 a Npt]"
    Npt=int(Ent_C_Npt.get());   Rmc=float(Ent_C_Rmc.get())
    L=int(Ent_C_L.get());       E=float(Ent_C_E.get())
    cri=Rmc/(Npt**2)  #factor conversion r=cri*i^2
    global r,P; Z=float(strZ.get())  #variables globales
    r=(Npt+1)*[0.];P=(Npt+1)*[0.] #r y P(r) para Ent_C_Npt puntos
    for i in range(1,Npt+1):r[i]=cri*i**2 #r[i] de 0 a Npt-1
    i_st=int(1+sqrt(L*(L+1)/(20*Z*cri))) #pto. mas prox. al origen valorado a efectos de escala y normalización
    P[Npt  ]=1000*exp(-sqrt(-2*E)*r[Npt  ]) #valores de partida
    P[Npt-1]=1000*exp(-sqrt(-2*E)*r[Npt-1])
    min_pr=max_pr=0
    if (P[Npt]==0.):
        messagebox.showwarning("Rmc too large","Please, try a smaller Rmc")

   #iteración Numerov
    for i in range(Npt-1,1,-1):
        r1=cri*(i-1)**2; r0=cri*i**2;r2=cri*(i+1)**2
        h1=r0-r1;h2=r2-r0
        hh1=h1*h1; hh2=h2*h2; h12=h1*h2; ha=h1/(h1+h2); hc=h2/(h1+h2)
        hD=hc*(hh1+h12-hh2)*Fpotencial(r1)/12.-hc
        hN=1+(hh1+3*h12+hh2)*Fpotencial(r0)/12.
        hM=ha-ha*(-hh1+h12+hh2)*Fpotencial(r2)/12. 
        P[i-1]=( hM*P[i+1]-hN*P[i] )/hD

   #normalización
    s=0.0
    for i in range(i_st,Npt+1):s+=i*P[i]**2
    s=sqrt(2*cri*s)  #integral /P(r)/^2dr
    for i in range(1,Npt+1):P[i]/=s
   #termina pintando resultado
    plotP()

def disables(): #disable todos los parámetros de potenciales
    Ent_Hdg_Z["state"]="disabled"
    Ent_SdR_Z["state"]="disabled";     Ent_SdR_A["state"]="disabled"
    Ent_SdR_a1["state"]="disabled";    Ent_SdR_a2["state"]="disabled"
    Ent_GSZ_Z["state"]="disabled";     Ent_GSZ_N["state"]="disabled"
    Ent_GSZ_H["state"]="disabled";     Ent_GSZ_D["state"]="disabled"

import matplotlib.pyplot as plt 
def plotP():
    plt.clf() #borra figura anterior
    #plt.close() #cierra ventana anterior
    Npt=int(Ent_C_Npt.get());   Rmc=float(Ent_C_Rmc.get())
    L=int(Ent_C_L.get());       Z=float(Ent_Hdg_Z.get())
    cri=Rmc/(Npt**2);           i_st=int(1+sqrt(L*(L+1)/(20*Z*cri)))
    plt.scatter(r[1:],P[1:],s=10)   #genera plot de los puntos P(r)
    plt.scatter(r[1:i_st],P[1:i_st],s=10) #marca los i_st primeros
    plt.xlabel('r'); plt.ylabel('P(r)') #pone título a los ejes
    plt.grid(True)
   #localización de extremos
    Npt=int(Ent_C_Npt.get())
    min_pr=P[Npt];max_pr=min_pr
    i_min=Npt;i_max=i_min
    for i in range(1,Npt+1):
        if (P[i]<min_pr):min_pr=P[i];i_min=i
        if (P[i]>max_pr):max_pr=P[i];i_max=i
    plt.text(0, 3,f'maximum P({r[i_max]:8.3})={P[i_max]:8.3}',transform=None)
    plt.text(0,15,f'minimum P({r[i_min]:8.3})={P[i_min]:8.3}',transform=None)
    plt.plot([0,Rmc],[0,0],color="c")
    plt.plot([0,0],[min_pr,max_pr],color="c") #marca los ejes por (0,0)
    #   plt.title("..."}
    plt.show()

#genera ventana con toda la información y opciones
W1=tkinter.Tk()
W1.geometry("410x400")
W1.title("Calculation of atomic orbitals for a Central Potential")
tkinter.Label(text="(C)F.Blanco 2022, Dpto. EMFTEL UCM").place(x=0,y=380)

BotHdg=tkinter.Button(text= "V Hydrogenic",command=runHdg).place(x=0,y=0)
tkinter.Label(text="Z=").place(x=0,y=35); #parámetro Z
Ent_Hdg_Z=tkinter.Entry();Ent_Hdg_Z.place(x=30,y=35)

BotSdR=tkinter.Button(text= "V Sanchez del Rio",command=runSdR).place(x=0,y=80)
tkinter.Label(text="Z=").place(x=0,y=115) #parámetro Z
Ent_SdR_Z=tkinter.Entry();Ent_SdR_Z.place(x=30,y=115)
tkinter.Label(text="A=").place(x=0,y=140) #parámetro A
Ent_SdR_A=tkinter.Entry();Ent_SdR_A.place(x=30,y=140)
tkinter.Label(text="a1=").place(x=0,y=165) #parámetro a1
Ent_SdR_a1=tkinter.Entry();Ent_SdR_a1.place(x=30,y=165)
tkinter.Label(text="a2=").place(x=0,y=190) #parámetro a2
Ent_SdR_a2=tkinter.Entry();Ent_SdR_a2.place(x=30,y=190)

BotSGZ=tkinter.Button(text= "V Green-Sellin-Zachor",command=runGSZ).place(x=0,y=235)
tkinter.Label(text="Z=").place(x=0,y=270) #parámetro Z
Ent_GSZ_Z=tkinter.Entry();Ent_GSZ_Z.place(x=30,y=270)
tkinter.Label(text="N=").place(x=0,y=295) #parámetro N
Ent_GSZ_N=tkinter.Entry();Ent_GSZ_N.place(x=30,y=295)
tkinter.Label(text="H=").place(x=0,y=320) #parámetro H
Ent_GSZ_H=tkinter.Entry();Ent_GSZ_H.place(x=30,y=320)
tkinter.Label(text="D=").place(x=0,y=345) #parámetro D
Ent_GSZ_D=tkinter.Entry();Ent_GSZ_D.place(x=30,y=345)

BotCal=tkinter.Button(text= "Refresh calculation",command=runCalc).place(x=250,y=0)
tkinter.Label(text="L=").place(x=250,y=35) #parámetro L
Ent_C_L=tkinter.Entry();Ent_C_L.place(x=280,y=35)
tkinter.Label(text="E=").place(x=250,y=60) #parámetro E
Ent_C_E=tkinter.Entry();Ent_C_E.place(x=280,y=60)
tkinter.Label(text="Rmc=").place(x=250,y=85) #parámetro Rmc
Ent_C_Rmc=tkinter.Entry();Ent_C_Rmc.place(x=280,y=85)
tkinter.Label(text="Npt=").place(x=250,y=110) #parámetro Npt
Ent_C_Npt=tkinter.Entry();Ent_C_Npt.place(x=280,y=110)

BotCal=tkinter.Button(text= "Save to file",command=runSave).place(x=250,y=155)
tkinter.Label(text="file:").place(x=250,y=190) #fichero grabar
Ent_F=tkinter.Entry();Ent_F.place(x=280,y=190)

#Inicializacion:
#variables definidas aquí para que sean globales
r=[0.];P=[0.]
strZ=tkinter.StringVar();strZ.set("1") #variable Z comun a los tres potenciales
Ent_Hdg_Z.config(textvariable=strZ)
Ent_GSZ_Z.config(textvariable=strZ)
Ent_SdR_Z.config(textvariable=strZ)
#activa sólo V Hdg
disables(); Ent_Hdg_Z["state"]="normal";
#inicializa para cálculo 1s del H
TipoV="Hdg" #Hdg,SdR,GSZ
Ent_C_L.insert(0,string="0");   Ent_C_E.insert(0,string="-0.5")
Ent_C_Rmc.insert(0,string="10");Ent_C_Npt.insert(0,string="300")
Ent_F.insert(0,string="orbital.txt")
#runCalc()#;plotP()

W1.mainloop()  #activa la ventana
