# On importe quelques outils pour les SARIMA, ACF/PACF, tests usuels, ...
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss, adfuller
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import datetime as dt
from datetime import datetime, timedelta


class Simulation():

    def __init__(self,n, mu, s2):
        self.n=n
        self.mu=mu
        self.s2=s2
        self.errors=np.random.normal(size=self.n)

    def produit_vectoriel(a, b):
        c=[]
        for i in range(len(a)):
            c.append(a[i]*b[i])
        return c


    def ARMA(self, Phi, Theta, choix):
        e=self.errors
        p=len(Phi)
        q=len(Theta)
        r=max(p,q)
        y=[0 for i in range(self.n)]
        Vcond=[self.s2 for i in range(self.n)]

        for i in range(self.n-r):
            if p>0:
                ar=sum(Simulation.produit_vectoriel(Phi,y[i:(i+p)]))
            else: ar=0 

            if q>0:
                ma=sum(Simulation.produit_vectoriel(Theta,e[i:(i+q)]))
            else: ma=0

            y[i]=ar + ma + e[i]

        y=[y[i]+self.mu for i in range(len(y))]
        
        if choix=="volatility":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,Vcond)
                plt.title("Volatilité")

        elif choix=="series":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,y)
                if len(Theta)!=0 and len(Phi)!=0:
                    plt.title(f"Time series : ARMA({len(Phi)},{len(Theta)})")
                elif len(Theta)!=0:
                    plt.title(f"Time series : MA({len(Theta)})")
                else:
                    plt.title(f"Time series : AR({len(Phi)})")
        
        elif choix=="comparaison":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,y)
                plt.title("Comparaison : ARMA-GARCH")

        elif choix == "values":
            return y, Vcond

    
    def GARCH(self, omega, alpha, beta, choix):
        e=self.errors
        r=max(len(alpha),len(beta))

        y=[0 for i in range(self.n)]
        t=np.zeros(self.n)

        variance=omega/(1-sum(alpha)-sum(beta))

        t[0]=np.random.normal()*(variance)**(1/2)
        ht=[0 for i in range(self.n)]
        for i in range(r,self.n):
            ht[i]=omega
            for a in range(len(alpha)):
                ht[i]+=alpha[a]*(t[i-(a+1)]**2)
            
            if len(beta)!=0:
                for b in range(len(beta)):
                    ht[i]+=beta[b]*(ht[i-(b+1)])
            
            t[i]=e[i]*ht[i]**(1/2)

            y[i]=t[i]

        y=[y[i]+ self.mu for i in range(len(y))]

        if choix=="volatility":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,ht)
                plt.title("Volatilité")

        elif choix=="series":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,y)
                if len(beta)!=0:
                    plt.title(f"Time series : GARCH({len(alpha)},{len(beta)})")
                else:
                    plt.title(f"Time series : ARCH({len(alpha)})")
        
        elif choix=="comparaison":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,y)
                plt.title("Comparaison : ARMA-GARCH")

        elif choix == "values":
            return y, ht

    
    def ARMA_GARCH(self, Phi, Theta, omega, alpha, beta, choix):
        e=self.errors
        p=len(Phi)
        q=len(Theta)
        r=max(p,q)

        R=max(len(alpha),len(beta))

        y=[0 for i in range(self.n)]
        t=np.zeros(self.n)
        variance=omega/(1-sum(alpha)-sum(beta))

        t[0]=np.random.normal()*(variance)**(1/2)
        ht=[0 for i in range(self.n)]
        for i in range(R,self.n):
            ht[i]=omega
            for a in range(len(alpha)):
                ht[i]+=alpha[a]*(t[i-(a+1)]**2)
            
            if len(beta)!=0:
                for b in range(len(beta)):
                    ht[i]+=beta[b]*(ht[i-(b+1)])
            
            t[i]=e[i]*ht[i]**(1/2)
        
        for i in range(self.n-r):
            if p>0:
                ar=sum(Simulation.produit_vectoriel(Phi,y[i:(i+p)]))
            else: ar=0 

            if q>0:
                ma=sum(Simulation.produit_vectoriel(Theta,t[i:(i+q)]))
            else: ma=0

            y[i]= ar + ma + t[i]
            
        y=[y[i]+self.mu for i in range(len(y))]

        if choix=="volatility":
            with plt.style.context("bmh"):
                fig = plt.figure(figsize=(12,5))
                x=[i for i in range(self.n)]
                plt.plot(x,ht)
                plt.title("Volatilité")

        elif choix=="series":
            fig = plt.figure(figsize=(12,5))
            with plt.style.context("bmh"):
                x=[i for i in range(self.n)]
                plt.plot(x,y)

                if len(Theta)!=0 and len(Phi)!=0:
                    if len(beta)!=0:
                        plt.title(f"Time series : ARMA({len(Phi)},{len(Theta)})-GARCH({len(alpha)},{len(beta)})")
                    else:
                        plt.title(f"Time series : ARMA({len(Phi)},{len(Theta)})-ARCH({len(alpha)})")
                
                elif len(Theta)!=0:
                    if len(beta)!=0:
                        plt.title(f"Time series : MA({len(Theta)})-GARCH({len(alpha)},{len(beta)})")
                    else:
                        plt.title(f"Time series : MA({len(Theta)})-ARCH({len(alpha)})")
        
                elif len(Phi)!=0:
                    if len(beta)!=0:
                        plt.title(f"Time series : AR({len(Phi)})-GARCH({len(alpha)},{len(beta)})")
                    else:
                        plt.title(f"Time series : AR({len(Phi)})-ARCH({len(alpha)})")

        elif choix == "values":
            return y, ht         

    def plot_ARMA(self,Phi, Theta):
        y,ht=Simulation.ARMA(self,Phi, Theta,"values")

        if len(Theta)!=0 and len(Phi)!=0:
            time_series=pd.DataFrame(y, columns=[f"ARMA({len(Phi)},{len(Theta)})"])
        elif len(Theta)!=0:
            time_series=pd.DataFrame(y, columns=[f"MA({len(Theta)})"])
        else:
            time_series=pd.DataFrame(y, columns=[f"AR({len(Phi)})"])
        
        volatility=pd.DataFrame(ht,columns=["volatility"])

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(10,8))
            layout = (3,2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0), alpha=0.5)
            pacf_ax = plt.subplot2grid(layout, (1, 1), alpha=0.5)
            volatility_ax = plt.subplot2grid(layout, (2, 0),colspan=2)
        
            time_series.plot(ax=ts_ax)
            if len(Theta)!=0 and len(Phi)!=0:
                ts_ax.set_title(f'Time Series : ARMA({len(Phi)},{len(Theta)})')
            elif len(Theta)!=0:
                ts_ax.set_title(f'Time Series : MA({len(Theta)})')
            else:
                ts_ax.set_title(f'Time Series : AR({len(Phi)})')

            plot_acf(time_series, ax=acf_ax)
            plot_pacf(time_series, ax=pacf_ax)      
            volatility.plot(ax=volatility_ax, color="red")
            volatility_ax.set_title("Volatility")

            plt.tight_layout()
    
    def plot_GARCH(self,omega, alpha, beta):
        y,ht=Simulation.GARCH(self,omega, alpha, beta,"values")
        
        if len(beta)!=0:
            time_series=pd.DataFrame(y, columns=[f"GARCH({len(alpha)},{len(beta)})"])
        else:
            time_series=pd.DataFrame(y, columns=[f"ARCH({len(alpha)})"])
        
        volatility=pd.DataFrame(ht,columns=["volatility"])

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(10,8))
            layout = (3,2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0), alpha=0.5)
            pacf_ax = plt.subplot2grid(layout, (1, 1), alpha=0.5)
            volatility_ax = plt.subplot2grid(layout, (2, 0),colspan=2)
        
            time_series.plot(ax=ts_ax)
            if len(beta)!=0:
                ts_ax.set_title(f'Time Series : GARCH({len(alpha)},{len(beta)})')
            else:
                ts_ax.set_title(f'Time Series : ARCH({len(alpha)})')
            
            plot_acf(time_series, ax=acf_ax)
            plot_pacf(time_series, ax=pacf_ax)      
            volatility.plot(ax=volatility_ax, color="red")
            volatility_ax.set_title("Volatility")

            plt.tight_layout()

    def plot_arma_garch_comparaison(self, Phi, Theta, omega, alpha, beta):
        ga,ht=Simulation.GARCH(self,omega, alpha, beta,"values")
        arma,vcond=Simulation.ARMA(self,Phi, Theta,"values")
        
        if len(beta)!=0:
            df_garch=pd.DataFrame(ga, columns=[f"GARCH({len(alpha)},{len(beta)})"])
        else:
            df_garch=pd.DataFrame(ga, columns=[f"ARCH({len(alpha)})"])

        df_arma=pd.DataFrame(arma, columns=[f"ARMA({len(Phi)},{len(Theta)})"])
        df_volatility=pd.DataFrame(ht,columns=["Variance"])
        df_vararma=pd.DataFrame(vcond,columns=["variance"])

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(12,8))
            layout = (2,2)
            arma_ax = plt.subplot2grid(layout, (0, 0))
            vararma_ax = plt.subplot2grid(layout, (1, 0))
            garch_ax = plt.subplot2grid(layout, (0, 1))
            volatility_ax = plt.subplot2grid(layout, (1, 1))

            df_arma.plot(ax=arma_ax)
            arma_ax.set_title(f'Time Series : ARMA({len(Phi)},{len(Theta)})')
            df_vararma.plot(ax=vararma_ax,color="red")
            vararma_ax.set_title(f'Variance : ARMA({len(Phi)},{len(Theta)})')
            df_garch.plot(ax=garch_ax)
            if len(beta)!=0:
                garch_ax.set_title(f'Time Series : GARCH({len(alpha)},{len(beta)})')
            else:
                garch_ax.set_title(f'Time Series : ARCH({len(alpha)})')
            
            df_volatility.plot(ax=volatility_ax, color="red")
            if len(beta)!=0:
                volatility_ax.set_title(f'Variance : GARCH({len(alpha)},{len(beta)})')
            else:
                volatility_ax.set_title(f'Variance : ARCH({len(alpha)})')


    def comparaison_arma_garch(self, Phi, Theta, omega, alpha, beta):
        arma_ga,ht_ga=Simulation.ARMA_GARCH(self,Phi, Theta, omega, alpha, beta,"values")
        ga,ht=Simulation.GARCH(self,omega, alpha, beta,"values")
        arma,vcond=Simulation.ARMA(self,Phi, Theta,"values")
        
        df_garch=pd.DataFrame(ga, columns=[f"GARCH({len(alpha)},{len(beta)})"])
        df_arma_garch=pd.DataFrame(arma_ga, columns=[f"ARMA({len(Phi)},{len(Theta)})-GARCH({len(alpha)},{len(beta)})"])
        df_arma=pd.DataFrame(arma, columns=[f"ARMA({len(Phi)},{len(Theta)})"])

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(12,8))
            layout = (2,1)
            arma_AG_ax = plt.subplot2grid(layout, (0, 0))
            garch_AG_ax = plt.subplot2grid(layout, (1, 0))

            df_arma_garch.plot(ax=arma_AG_ax)
            df_arma.plot(ax=arma_AG_ax)
            arma_AG_ax.set_title(f'Comparaison : ARMA({len(Phi)},{len(Theta)}) et ARMA({len(Phi)},{len(Theta)})-GARCH({len(alpha)},{len(beta)})')

            df_arma_garch.plot(ax=garch_AG_ax)
            df_garch.plot(ax=garch_AG_ax)
            garch_AG_ax.set_title(f'Comparaison : GARCH({len(alpha)},{len(beta)}) et ARMA({len(Phi)},{len(Theta)})-GARCH({len(alpha)},{len(beta)})')

    
    def plot_qqplot_garch(self,omega, alpha, beta):
        y,ht=Simulation.GARCH(self,omega, alpha, beta,"values")
        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(10,5))
            layout = (1,1)
            qqplot_ax = plt.subplot2grid(layout, (0, 0))
            scs.probplot(y, sparams=(np.mean(y), np.std(y)), plot=qqplot_ax)
            plt.title("QQ Plot")


    def superposition(self):
        ga,ht=Simulation.GARCH(self,1, [0.5], [],"values")
        arma,vcond=Simulation.ARMA(self,[0.5], [],"values")
        df2=pd.DataFrame(ga, columns=[f"ARCH(1)"])
        df2[f"AR(1)"]=arma 

        ga,ht=Simulation.GARCH(self,1, [0.5], [0.3],"values")
        arma,vcond=Simulation.ARMA(self,[0.5], [0.3],"values")
        df=pd.DataFrame(ga, columns=[f"GARCH(1,1)"])
        df[f"ARMA(1,1)"]=arma 

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(12,8))
            layout = (2,1)
            ar_arch_ax= plt.subplot2grid(layout, (0, 0), colspan=2)
            arma_garch_ax = plt.subplot2grid(layout, (1, 0), colspan=2)

            df2.plot(ax=ar_arch_ax)
            ar_arch_ax.set_title("Superposition : AR(1) - ARCH(1)")
            df.plot(ax=arma_garch_ax)
            arma_garch_ax.set_title("Superposition : ARMA(1,1) - GARCH(1,1)")

        
    def plot_GARCH_carre(self,omega, alpha, beta):
        y,ht=Simulation.GARCH(self,omega, alpha, beta,"values")
        
        if len(beta)!=0:
            time_series=pd.DataFrame(y, columns=[f"GARCH({len(alpha)},{len(beta)})"])
        else:
            time_series=pd.DataFrame(y, columns=[f"ARCH({len(alpha)})"])
        
        volatility=pd.DataFrame(ht,columns=["volatility"])

        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(10,8))
            layout = (3,2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0), alpha=0.5)
            pacf_ax = plt.subplot2grid(layout, (1, 1), alpha=0.5)
            volatility_ax = plt.subplot2grid(layout, (2, 0),colspan=2)
        
            time_series.plot(ax=ts_ax)
            if len(beta)!=0:
                ts_ax.set_title(f'Time Series : GARCH({len(alpha)},{len(beta)})')
            else:
                ts_ax.set_title(f'Time Series : ARCH({len(alpha)})')
            
            plot_acf(time_series**2, ax=acf_ax)
            plot_pacf(time_series**2, ax=pacf_ax)      
            volatility.plot(ax=volatility_ax, color="red")
            volatility_ax.set_title("Volatility")

            plt.tight_layout()


class Prevision():
    def __init__(self, df, p, q):
        self.df=df
        self.p=p
        self.q=q
    
    def forecast(self,last_obs):
        mod = arch_model(self.df, vol="Garch", p=self.p, q=self.q)
        res = mod.fit(disp="off")
        mod_train = arch_model(self.df, vol="Garch", p=self.p, q=self.q)
        res_train = mod_train.fit(disp="off", last_obs=last_obs)

        format_str = "%Y-%m-%d"
        date = datetime.strptime(last_obs, format_str)

        day_start = date + timedelta(days=1)

        forecasts = res_train.forecast(start=day_start.strftime(format_str), reindex=False)
        cond_var = forecasts.variance
        
        cond_var=cond_var.rename(columns={"h.1":"Python"})
        y_true=pd.DataFrame(res.conditional_volatility[-len(cond_var):])
        y_true=y_true.rename(columns={"cond_vol":"Serie"})
        return cond_var, y_true
    
    def forecast_calcul(self,last_obs):
        cond_var, y_true =Prevision.forecast(self,last_obs)

        mod = arch_model(self.df, vol="Garch", p=self.p, q=self.q)
        res = mod.fit(disp="off")

        omega=res.params[1]
        predict=[]

        for i in range(len(cond_var)):
            pred=omega
            for j in range(self.p):
                pred+= res.params[j+2]*(self.df[(-len(cond_var))+i-j])**2 
            
            if self.q !=0 :
                for k in range(self.q):
                    pred+= res.params[-(k+1)]*(res.conditional_volatility[(-len(cond_var))+i-k])**2
                    
            predict.append(pred)

        cond_var["Calculer"]=predict
        y_true=pd.DataFrame(res.conditional_volatility[-len(cond_var):])
        y_true=y_true.rename(columns={"cond_vol":"Serie"})

        return cond_var, y_true

