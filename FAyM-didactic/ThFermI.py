# Solucion del modelo Thomas Fermi para átomos o iones positivos
# versión basic: F. Sanchez Ferrero 1991
# versiones QB: F. Blanco 1992-2000-2005
# versión C: F. Blanco 2016
# versión python: F. Blanco 2022

import matplotlib.pyplot as plt
from math import sqrt,exp

def CalcXi(): #cálculo con x0 introducida
    if Ent_xi.get()=="":return
    global x0;x0=float(Ent_xi.get());calcFi()

def CalcXs(): #cálculo con x0 sugerida
    global x0;x0=float(Ent_xs.get());calcFi()

def calcFi(): #resuelve numéricamente Fi()
    global N,Z,npt,x0,X,Fi,xs #variables cálculo numérico
    global xo1,yo1,xo2,yo2 #anota mejores intentos de x0 para estimar xs
	#Tareas en la primera llamada
    if len(X)==1:
       #chequeo previo de datos
        Z=int(Ent_Z.get())
        if Z<1 or Z>100 :return
        N=int(Ent_N.get())
        if N<1 or N>Z :return
        npt=int(Ent_npt.get())
        if npt<10 or npt>5000 :return
       #desactivamos entrada parámetros básicos
        Ent_Z["state"]="disabled";Ent_N["state"]="disabled";Ent_npt["state"]="disabled";
       #y activo botón "valid" puesto que ya va a haber datos qu validar
        BotValid["state"]="normal"
       #inicializa X,Fi
        if len(X)==1:
            for i in range (1,npt+1):X.append(0);Fi.append(0) #npt ptos + origen
   #realizo cálculo
    H=x0/npt;H12=H*H/12.
    if N==Z: #caso átomo neutro, Fi analítica
        for i in range(0,npt+1):
            X[i]=H*i;            sx=sqrt(X[i]);      ifi=0.006944
            ifi=ifi*sx+0.007298; ifi=ifi*sx+0.2302;  ifi=ifi*sx-0.1486
            ifi=ifi*sx+1.243;    ifi=ifi*sx+0.02747; ifi=ifi*sx+1.
            Fi[i]=1./ifi
    else: #caso ión, cálculo numérico
        for i in range(0,npt+1):X[i]=H*i;
        Fi[npt]=0.; Fi[npt-1]=(float(Z-N)/Z)*H/x0;
        for i in range(npt-2,0,-1):
            A=2.*( 1.+ 5.*H12*sqrt( Fi[i+1]/X[i+1] ) )
            B=     1.-    H12*sqrt( Fi[i+2]/X[i+2] ) 
            Fi[i]=2*Fi[i+1]-Fi[i+2] #estimación de partida
            while 1:                #itera la estimación de Fi[i-1] para numerov
                Fest=Fi[i]
                C= 1.-    H12*sqrt( Fest   /X[i] )
                Fi[i]=( A*Fi[i+1] - B*Fi[i+2] )/C
                if ( (Fi[i]/Fest-1)<1.e-6 ): break  #hasta autoconsistencia
        Fi[0]=2.*Fi[1]+H*H*Fi[1]*sqrt(Fi[1]/X[1])-Fi[2] #extrapola Fi[0]
   #Ya tenemos la Fi, evaluo integral carga como chequeo
    IntRho=0.
    for i in range(1,npt+1):IntRho +=Fi[i]**1.5 *X[i]**0.5
    IntRho+= H**.5*(19+9*Fi[1])/60 #correccion por integral en el primer dx
    IntRho*=Z*H
   #estima x0 recomendado
    if N==Z: #caso átomo neutro, Fi analítica
        xs=30
    else: #caso ión, cálculo numérico
        if xo1==-1.:              #si es el primer valor disponible...
            xo1=x0;yo1=Fi[0]         #lo anota
            xs/=Fi[0]                #y da una primera sugerencia de cambio
        else:
            if xo2!=-1.:          #si ya tenemos dos valores toma el mejor
                if abs(yo2-1.)<abs(yo1-1.):tmp=xo1;xo1=xo2;xo2=tmp;tmp=yo1;yo1=yo2;yo2=tmp
                if abs(Fi[0]-1.)<abs(yo2-1.):xo2=x0;yo2=Fi[0]
            else:                 #si es el segundo valor disponible
                if x0!=xo1: xo2=x0;yo2=Fi[0]
            if yo1!=yo2:
                xs=xo1-(yo1-1)*(xo2-xo1)/(yo2-yo1)
    Ent_xs.delete(0,"end");Ent_xs.insert(0,string=str(xs))
   #muestro resultados
    tkinter.Label(text="Checks for the obtained solution").place(x=0,y=240)
    tkinter.Label(text="  Fi(0)="+f"{Fi[0]:<8.3}"+"       ").place(x=0,y=260)
    tkinter.Label(text="  Integrated charge="+f"{IntRho:<8.3}"+"       ").place(x=0,y=280)
    tkinter.Label(text="If checks were not O.K., perhaps you should...").place(x=0,y=300)
    if(N==Z): tkinter.Label(text="- try with a larger maximum radius").place(x=0,y=320)
    if(N< Z): tkinter.Label(text="- try another x0 starting point").place(x=0,y=320)
    tkinter.Label(text="- try with a larger number of points").place(x=0,y=340)
   #plot Fi
    plt.clf() #borra figura anterior [no sé aquí si será preferible .clf() o .cla() ]
    plt.title("Resulting Fi function")
    plt.plot(X,Fi,marker=".", linestyle=":")
    plt.plot([0,x0/30],[1,1],color="g") #marca el pt. (0,1) donde deberia corta Fi
    plt.plot([0,0],[0,1.3],color="c") #marca los ejes por (0,0)
    plt.plot([0,x0],[0,0],color="c")
    plt.xlabel('x'); plt.ylabel('Fi(x)') #pone título a los ejes
    plt.grid(True)
    plt.ylim(0,1.3);plt.xlim(0,x0) #intervalo ejes
    plt.text(0, 3,f'Fi(0)={Fi[0]:8.3}           x0={x0:8.5}',transform=None)
    plt.show()

from datetime import datetime #manipulación de fechas y horas
def Accept(): #solucion aceptada, mostrar resultados
   #desactivamos entrada parámetros y activamos nombre fichero y botón save
    Ent_xi["state"]="disabled";Ent_xs["state"]="disabled"
    Ent_F["state"]="normal";BotSave["state"]="normal";BotValid["state"]="disabled"
   #preparo R, V y rho
    global npt,X,Fi,N,Z,x0,R,Rho,Zef,Ef,K
    K=.88534/Z**(1/3) ; Ef=-(Z-N)/(K*x0)  #r=Kx, EFermi
    tmp=Z/(4.*3.141593*K**3);r0=K*x0
    for i in range(1,npt+1):
        R.append(X[i]*K)
        Rho.append(tmp*(Fi[i]/X[i])**1.5)
        Zef.append(Z*Fi[i]-Ef*(K*X[i])) #Zeff=-r*V(r)
   #cierra plot y prepara otro con más gráficas
    plt.close();ventana=plt.figure("Summary of results")
    plt.figtext(0.6,0.3,f'x0={x0:8.5} \nr0={x0*K:8.5}')
   #Plot Fi como el 1o de 2x2
    Pl_Fi=ventana.add_subplot(221)#;subplot('Position',[0.1 0.3 0.2 0.2])
    plt.title("Fi(x) function");plt.xlabel('x')
    plt.xlim(0,x0);plt.ylim(0,1)
    plt.plot([0,0],[0,1],color="c");plt.plot([0,x0],[0,0],color="c") #ejes por (0,0)
    linea1,=Pl_Fi.plot(X,Fi,marker=".", linestyle=":")
   #Plot rho como el 2o de 2x2
    Pl_rho=ventana.add_subplot(222)
    plt.title("log of rho(r) charge density");plt.xlabel('r');
    plt.yscale("log");plt.xlim(0,K*x0) 
    linea2,=Pl_rho.plot(R[1:],Rho[1:],marker=".", linestyle=":")
   #Plot Zeff como el 3o de 2x2
    Pl_Zef=ventana.add_subplot(223) #,position=[.2,.1,0.4,0.2])
    plt.xlabel('r');plt.ylabel("Zeff=-r*V(r)");plt.title(" ") # plt.title("Zeff=-r*V(r)")
    plt.ylim(0,Z);plt.xlim(0,K*x0)
    plt.plot([0,0],[0,Z],color="c");plt.plot([0,x0*K],[0,0],color="c") #ejes por (0,0)
    linea1,=Pl_Zef.plot(R[1:],Zef[1:],marker=".", linestyle=":")
    plt.show()
    
def FileSave(): #grabar en fichero
    global npt,X,Fi,N,Z,x0,R,Rho,Vr,Ef,K
    f=open(Ent_F.get(),"w")
    f.write( datetime.now().strftime(' %Y-%m-%d %H:%M:%S       ') )
    f.write("Solution of the Thomas Fermi model for atoms"+"\n")
    f.write(" Z="+str(Z)+"   N="+str(N)+"   N.Points="+str(npt) )
    f.write(f"   x0={x0:<6.4}  r0={K*x0:< 6.4}  K={K:< 6.4}  Ef={Ef:< 6.4}\n")
    f.write(" i     x           Fi          r           rho          V\n")
    f.write(f"{0: 4}  {X[0]:< 5.3e}  {Fi[0]:< 5.3e}  {K*X[0]:< 5.3e}   inf         -inf\n")
    for i in range(1,npt+1):
        f.write(f"{i: 4}  {X[i]:< 5.3e}  {Fi[i]:< 5.3e}  {K*X[i]:< 5.3e}  {Rho[i]:< 5.3e}   {-Zef[i]/(K*X[i]):< 5.3e}\n")
    f.close
   #cierra el programa
    plt.close();WIO.destroy()
    import sys #para usar la funcion de salida sys.exit()
    sys.exit() # tanto quit() como #exit() dan problema en alguna version
    	
	
#genera ventana con toda la información y menús
import tkinter #importa módulo de ventanas y botones
WIO=tkinter.Tk()
WIO.geometry("600x400")
WIO.title("Solving the Thomas-Fermi model for atoms")
tkinter.Label(text="(C)F.Blanco 2022, Dpto. EMFTEL UCM").place(x=0,y=375)
#datos de entrada
tkinter.Label(text="Basic parameters").place(x=0,y=5)
tkinter.Label(text="nuclear Z=").place(x=0,y=25) #Z átomo
Ent_Z=tkinter.Entry(width=8);Ent_Z.place(x=95,y=25)
tkinter.Label(text="N electrons=").place(x=0,y=45) #N electrones
Ent_N=tkinter.Entry(width=8);Ent_N.place(x=95,y=45)
tkinter.Label(text="no of calculated points=").place(x=0,y=65) #número ptos cálculo
Ent_npt=tkinter.Entry(width=8);Ent_npt.place(x=170,y=65);Ent_npt.insert(0,string="500")
#estimación x0
tkinter.Label(text="Estimation of maximun radius x0").place(x=0,y=100)
tkinter.Label(text="your estimation x0=").place(x=0,y=120) #x0 introducido
Ent_xi=tkinter.Entry(width=12);Ent_xi.place(x=140,y=120)
tkinter.Label(text="suggested x0=").place(x=0,y=140) #x0 sugerido
Ent_xs=tkinter.Entry(width=12);Ent_xs.place(x=140,y=140);Ent_xs.insert(0,string="1")
#botones de cálculo usando x0 introducido/sugerido
BotCalcXi=tkinter.Button(text= "Compute Fi",command=CalcXi).place(x=245,y=110)
BotCalcXs=tkinter.Button(text= "Compute Fi",command=CalcXs).place(x=245,y=140)
BotValid=tkinter.Button(text= "Accept\nsolution",command=Accept)
BotValid.place(x=340,y=112);BotValid["state"]="disabled"
#guardar resultados o.k.
BotSave=tkinter.Button(text= "Save results \n and exit",command=FileSave)
BotSave.place(x=0,y=185);BotSave["state"]="disabled"
tkinter.Label(text="file:").place(x=100,y=200) #fichero grabar
Ent_F=tkinter.Entry(width=30);Ent_F.place(x=130,y=200)
Ent_F.insert(0,string="ThFermi.txt");Ent_F["state"]="disabled";
#Inicializo aquí variables que serán globales:
X=[0];Fi=[0];Z=1;N=1;npt=500;xs=1  #para el cálculo numérico
R=[0];Rho=[0];Zef=[0];Ef=0;K=0     #para los resultados
xo1=-1.;yo1=0.;xo2=-1.;yo2=0. #anotará los mejores intentos de x0 para estimar xs

WIO.mainloop()  #activa la ventana
