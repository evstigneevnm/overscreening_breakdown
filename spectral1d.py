import numpy as np
from scipy import optimize, integrate
from matplotlib import pyplot as plt
import pickle
import mpmath #for custom pecision arithmetic
from scipy.optimize import minimize
from scipy.integrate import ode

class basis_functions(object):
    def __init__(self, N, L = 1, use_mpmath = False, prec = 100, calc_domain = [0, np.inf]):
        self.__N = N
        self.__M = N+2 #for boundary conditions
        self.__L = L
        self.__use_mpmath = use_mpmath
        if self.__use_mpmath:
            self.__mp = mpmath
            self.__mp.mp.dps = prec
        else:
            self.__mp = np
        self._domain = (0,np.pi)
        self._calc_domain = calc_domain
        self.__cos = np.vectorize(self.__mp.cos)
        self.__sin = np.vectorize(self.__mp.sin)
        if self.__use_mpmath:
            self.__atan = np.vectorize(self.__mp.atan)
            self.__acos = np.vectorize(self.__mp.acos)
        else:
            self.__atan = np.vectorize(self.__mp.arctan)
            self.__acos = np.vectorize(self.__mp.arccos)
        self.__sqrt = np.vectorize(self.__mp.sqrt)
        self.__tan = np.vectorize(self.__mp.tan)


    #axilary functions

    def __convert(self, val):
        if self.__use_mpmath:
            return self.__mp.mpf(1)*val
        else:
            return val

    def __get_length(self, var):
        try:
            size_var = len(var)
        except:
            size_var = 1
        return size_var
    def use_mpmath(self, prec = 100):
        self.__use_mpmath = True
        self.__mp = mpmath
        self.__mp.mp.dps = prec

    def use_numpy(self):
        self.__use_mpmath = False
        self.__mp = np

    def get_mpmath_ref(self):
        return self.__mp

    def is_using_mpmath(self):
        return self.__use_mpmath
    def __check(self, val):
        try:
            len(val)
            return(val) 
        except:
            return([val])

    # t \in [0, Pi]
    def psi(self, k, t):
        return(self.__cos( self.__check(k*t) ) ) 
    def dpsi(self, k, t):        
        return(-k*self.__sin(self.__check(k*t) ))
    def ddpsi(self, k, t):
        return(-k*k*self.__cos(self.__check(k*t) ))
    def dddpsi(self, k, t):      
        return(k*k*k*self.__sin(self.__check(k*t) ))
    def ddddpsi(self, k, t):  
        return(k*k*k*k*self.__cos(self.__check(k*t) ))

    def circlemap_in_t(self, j):
        if (np.max(j)>self.__N) or (np.min(j)<0):
            raise ValueError("logical_error: __circlemap_in_t j: "+str(np.max(j)) + " " + str(np.min(j)) );
        return self.__mp.pi*(2*(j+1) - 1)/(2.0*self.__N)
    
    def circlemap_with_bound_in_t(self, j): 
    #it is assumed that j runs through 0,...,self._M-1 while Chebyshev zeroes are located at k(=j-1) in 0,...,self.N_-1
        k = j-1;
        if (np.max(j)>self.__M) or (np.min(j)<0):
            raise ValueError("logical_error: circlemap_with_bound_in_t j: "+str(np.max(k)) + " " + str(np.min(k)) );
        j_length = self.__get_length(j)
        if j_length == 1:
            if j==0:
                value = 0;
            elif j==self.__M-1:
                value = self.__mp.pi
            else:
                value = self.__mp.pi*(2*(k+1) - 1)/(2*self.__N)
        else:
            value = self.__mp.pi*(2*(k+1) - 1)/(2*self.__N)
            value[0] = 0;
            value[-1:] = self.__mp.pi

        return value

    # def from_infinity_map_to_t_old(self, x_in):
    #     x = x_in.copy()
    #     zero_at = np.argmin(x)
    #     if x[zero_at]==0:
    #         x[zero_at] = 1
    #         print(x)
    #         y = 1/x
    #         val = 2*self.__atan(self.__sqrt(self.__L*y))
    #         val[zero_at] = self.__mp.pi
    #     else:
    #         y = 1/x
    #         val = 2*self.__atan(self.__sqrt(self.__L*y))
    #     return val

    def from_infinity_map_to_t(self, x_in): #not good, but have to do it due to mpmath
        x = x_in.copy()
        xind = (np.float128(x) == 0)
        for jl, xl in enumerate(x):
            if xl == 0:
                x[jl] = 1
        y = 1/x
        val = 2*self.__atan(self.__sqrt(self.__L*y))
        for jl, fl in enumerate(xind):
            if fl:
                val[jl] = self.__mp.pi
        return val

    def from_segment_map_to_t(self, x_in): #not good, but have to do it due to mpmath
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        idx0 = x_in < x0
        idx1 = x_in > x1
        for jl, fl in enumerate(idx0):
            if fl:
                x_in[jl] = x0
        for jl, fl in enumerate(idx1):
            if fl:
                x_in[jl] = x1

        t = self.__acos((2*x_in-(x1 + x0))/(x1 - x0))
        return t    

    def from_t_map_to_infinity(self, t):
        ss = self.__sin(t/2)**2;
        cs = self.__cos(t/2)**2;
        inf_at = np.argmin(ss)
        if self.__get_length(t)>1:
            if ss[inf_at]==0:
                ss[inf_at] = 1
                cs[inf_at] = self.__mp.inf
        else:
            if ss == 0:
                ss = 1
                cs = self.__mp.inf
        return self.__L*cs/ss;

    def from_t_map_to_segment(self, t):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        x = (self.__cos(t)*(x1-x0)+(x1+x0))/2
        return x

    def map_derivative1_inf(self, l, t):
        return -self.__sin(t/2)**2*self.__tan(t/2)*self.dpsi(l, t)/self.__L
    def map_derivative2_inf(self, l, t):
        return 1/(2*self.__L**2)*self.__sin(t/2)**2*self.__tan(t/2)**3*((2 + self.__cos(t))*self.dpsi(l, t) + self.__sin(t)*self.ddpsi(l, t))
    def map_derivative3_inf(self, l, t):
        common =  -1/(4*self.__L**3)*self.__sin(t/2)**2*self.__tan(t/2)**5
        return common*((8 + 6*self.__cos(t) + self.__cos(2*t))*self.dpsi(l, t) + self.__sin(t)*(3*(2 + self.__cos(t))*self.ddpsi(l, t)+self.__sin(t)*self.dddpsi(l, t)));
    def map_derivative4_inf(self, l, t):
        common = 1/(16*self.__L**4)*self.__sin(t/2)**2*self.__tan(t/2)**7;
        p1 = 3*(32 + 29*self.__cos(t) + 8*self.__cos(2*t) + self.__cos(3*t))*self.dpsi(l, t);
        p2 = (91 + 72*self.__cos(t) + 11*self.__cos(2*t))*self.ddpsi(l, t);
        p3 = (6*(2 + self.__cos(t))*self.dddpsi(l, t) + self.__sin(t)*self.ddddpsi(l, t));
        return common*(p1+self.__sin(t)*(p2 + 2*self.__sin(t)*p3));
    

    def map_derivative1_segment(self, l, t):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return (2*1/self.__sin(t)*self.dpsi(l, t))/(x0-x1)
    def map_derivative2_segment(self, l, t):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return -((4*(1/(self.__sin(t))**2)*(self.__cos(t)/self.__sin(t)*self.dpsi(l, t)-self.ddpsi(l, t)))/(x0-x1)**2)
    def map_derivative3_segment(self, l, t):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        #(self.__mp.pi**3)*self.dddpsi(l, t)/((x1-x0)**3)
        p0 = 8*(1/self.__sin(t)**3)
        p1 = (-2+3/self.__sin(t)**2)*self.dpsi(l, t)
        p2 = -3*(self.__cos(t)/self.__sin(t))*self.ddpsi(l, t)+self.dddpsi(l, t)
        return p0*(p1+p2)/(x0-x1)**3
    def map_derivative4_segment(self, l, t):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        #(self.__mp.pi**4)*self.ddddpsi(l, t)/((x1-x0)**4)    
        p0 = -(1/(x0-x1)**4)*16*(1/self.__sin(t)**4)
        p1 = (6*(self.__cos(t)/self.__sin(t))**3+9*(self.__cos(t)/self.__sin(t))*1/self.__sin(t)**2)*self.dpsi(l, t)
        p2 = (11-15/self.__sin(t)**2)*self.ddpsi(l, t)
        p3 = 6*self.__cos(t)/self.__sin(t)*self.dddpsi(l, t) - self.ddddpsi(l, t)
        return p0*(p1+p2+p3)



    #Assuming[{k > 0, k \[Element] Integers}, Limit[MapDerivative1[k, t], t -> Pi]]
    def map_value_derivative1_inf_at_pi(self, k):
        return -(2/self.__L)*((-1)**k)*(k**2)
    def map_value_derivative2_inf_at_pi(self, k):
        return (4/(3*self.__L**2))*((-1)**k)*(k**2)*(2 + k**2)      
    def map_value_derivative3_inf_at_pi(self, k):
        return -(4/(15*self.__L**3))*((-1)**k)*(k**2)*(23 + 20*k**2 + 2*k**4)
    def map_value_derivative4_inf_at_pi(self, k):
        return 16/(105*self.__L**4)*((-1)**k)*(k**2)*(132+154*k**2+28*k**4+k**6)


    # 0->x1, pi->x0
    def map_derivative1_segment_at_0(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]        
        return -((2*n**2)/(x0-x1))
    def map_derivative2_segment_at_0(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return ((4*n**2)*(n**2-1))/(3*(x0-x1)**2)
    def map_derivative3_segment_at_0(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]        
        return -(( (8*n**2)*(4-5*n**2+n**4))/(15*(x0-x1)**3))
    def map_derivative4_segment_at_0(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return ((16*n**2)*(-36+49*n**2-14*n**4+n**6))/(105*(x0-x1)**4)


    def map_derivative1_segment_at_pi(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1] 
        return (2*((-1)**n)*n**2)/(x0-x1)
    def map_derivative2_segment_at_pi(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return ((4*((-1)**n)*n**2)*(n**2-1))/(3*(x0-x1)**2)
    def map_derivative3_segment_at_pi(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]        
        return ((8*((-1)**n)*n**2)*(4-5*n**2+n**4))/(15*(x0-x1)**3)
    def map_derivative4_segment_at_pi(self, n):
        x0 = self._calc_domain[0]
        x1 = self._calc_domain[1]
        return ((16*((-1)**n)*n**2)*(-36+49*n**2-14*n**4+n**6))/(105*(x0-x1)**4)


    def set_mapping_parameter(self, value):
        self.__L=value

    def test_basis_functions(self):
        #all these values are fixed from wolfram mathematica for basis function verification
        L_save = self.__L
        self.__L = 1;
        point_1 = 0.5513288954217920495113264983129694413973864803666406527993660202910303034692697948403800288231708892;
        k = np.arange(0,10)
        k = self.__convert(k)
        ref_psi = [1.000000000000000000000000, 0.8518291728351008482475919, 0.4512258793858642257917624, -0.08309443763699743507355421, -0.5927904115449070198720477, -0.9268178942247568910640156, -0.9861906288675822882369444, -0.7533140010672442067997559, -0.2971990559608395323392074, 0.2469883490542546328224461]
        res_psi = self.psi(k,point_1)
        if np.linalg.norm(ref_psi - res_psi)/np.linalg.norm(ref_psi)>1.0e-12:
            raise ValueError("test_basis_functions: psi(k,point_1)" )
        ref_dpsi = [0, -0.5238196830084450798539843, -1.784819549167314011954522,-2.989625031655996871675229, -3.221427082472411986522575, -1.877555531434339141698478, 0.9936848430037828588887108, 4.603627132382350067000330, 7.638524343922920915588764, 8.721166045314568349642232]
        res_dpsi = self.dpsi(k,point_1)
        if np.linalg.norm(ref_dpsi - res_dpsi)/np.linalg.norm(ref_dpsi)>1.0e-12:
            raise ValueError("test_basis_functions: dpsi(k,point_1)" )
        ref_ddpsi = [0, -0.8518291728351008482475919, -1.804903517543456903167050, 0.7478499387329769156619879, 9.484646584718512317952763, 23.17044735561892227660039, 35.50286263923296237653000, 36.91238605229496613318804, 19.02073958149373006970927,-20.00605627339462525861814]
        res_ddpsi = self.ddpsi(k, point_1)
        if np.linalg.norm(ref_ddpsi - res_ddpsi)/np.linalg.norm(ref_ddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: ddpsi(k,point_1)" )
        ref_dddpsi = [0, 0.5238196830084450798539843, 7.139278196669256047818088,26.90662528490397184507706, 51.54283331955859178436120,46.93888828585847854246195, -35.77265434813618291999359,-225.5777294867351532830162, -488.8655580110669385976809,-706.4144496704800363210208]
        res_dddpsi = self.dddpsi(k, point_1)
        if np.linalg.norm(ref_dddpsi - res_dddpsi)/np.linalg.norm(ref_dddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: res_dddpsi(k,point_1)" )
        ref_ddddpsi = [0, 0.8518291728351008482475919, 7.219614070173827612668199,-6.730649448596792240957891, -151.7543453554961970872442,-579.2611838904730569150098, -1278.103055012386645555080,-1808.706916562453340526214, -1217.327333215598724461394,1620.490558144964645948069]
        res_ddddpsi = self.ddddpsi(k, point_1)
        errdddd = np.linalg.norm(res_ddddpsi - ref_ddddpsi)/np.linalg.norm(ref_ddddpsi)
        if errdddd>1.0e-12:
            raise ValueError("test_basis_functions: ref_ddddpsi(k,point_1): " + str(errdddd) )
        
        ref_map_dpsi = [0, 0.01097729701136520820034867, 0.03740312733262579974626982,0.06265133401749413623782799, 0.06750903608978212255593176,0.03934652589897829732072724, -0.02082390946956544989825558,-0.09647476794209319156219345, -0.1600748371466477000020034,-0.1827629489121505488706909]
        res_map_dpsi = self.map_derivative1_inf(k, point_1)
        if np.linalg.norm(ref_map_dpsi - res_map_dpsi)/np.linalg.norm(ref_map_dpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative1_inf(k,point_1)" )
        ref_map_ddpsi = [0, -0.001626515178208758263781210, -0.005060048116726302672405319,-0.006819588557051617330710577, -0.003536922907775971605573123,0.005686512491409826468878889, 0.01796743812094209954558021,0.02721761050982603228438492, 0.02661649575949853008326341,0.01206584604865973854736720]
        res_map_ddpsi = self.map_derivative2_inf(k, point_1)
        if np.linalg.norm(ref_map_ddpsi - res_map_ddpsi)/np.linalg.norm(ref_map_ddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative2_inf(k,point_1)" )
        ref_map_dddpsi = [0, 0.0003615031490271825956798807, 0.001017498831196142756221112,0.0009999119467623338405112287, -0.0004346522687072531548097935,-0.003060784063570530240319312, -0.005459424330207586721835725,-0.005566605937243231044743460, -0.001834660285750301881418712,0.005541342830452007071041159]
        res_map_dddpsi = self.map_derivative3_inf(k, point_1)
        if np.linalg.norm(ref_map_dddpsi - res_map_dddpsi)/np.linalg.norm(ref_map_dddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative3_inf(k,point_1)" )
        ref_map_ddddpsi = [0, -0.0001071284412281469066246027, -0.0002697806674159795073070278,-0.0001528750722007874915594755, 0.0004292431812431379781686090,0.001237269907970965895016242, 0.001611226807497628257903866,0.0008286554393752993557871492, -0.001337174375495872372237177,-0.004186629741370010576532872]
        res_map_ddddpsi = self.map_derivative4_inf(k, point_1)
        if np.linalg.norm(ref_map_ddddpsi - res_map_ddddpsi)/np.linalg.norm(ref_map_ddddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative4_inf(k,point_1)" )
        self.__L = L_save
        return(self)



    def test_basis_functions_segment(self):
        calc_domain_back = self._calc_domain.copy()
        self._calc_domain = [-4, 7]
        point_1 = 0.5513288954217920495113264983129694413973864803666406527993660202910303034692697948403800288231708892;
        k = np.arange(0,10)
        k = self.__convert(k)
        ref_map_dpsi = [0,0.1818181818181818181818181818181818181818181818181818181818181818181818181818181818181818181818181818,0.619512125698255162361885049417263711454192437720704873280726486776741765366305344054658804917952625,1.037700959330033700863740839830479350835234083574345601260019310441272684577101696944947207146834973,1.118159614833604964616782158753638277215719958890586544799333433264753649588120545130535730340148069,0.651700850619922192581299533634451134989323933862139639383142580958796778233186737888431559411409945,-0.34490871060360758812358807101490200844245777514501424278268129806437711994275857366266420418380965,-1.59792222806777293680749360272444743553543677938009773517782953386117465867533726912105814863775125,-2.65133719300042962818831651285264959811814640174953019075440607209737852554140339834291466349719579,-3.02712289195897510251197573984911823111227928590241381145194753216153001870123981514156798004734770]
        res_map_dpsi = self.map_derivative1_segment(k, point_1)
        if np.linalg.norm(ref_map_dpsi - res_map_dpsi)/np.linalg.norm(ref_map_dpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative1_segment(k,point_1)" )        

        ref_map_ddpsi = [0,0,0.132231404958677685950413223140495867768595041322314049586776859504132231404958677685950413223140496,0.675831409852641995303874599364287685222755386604405316306247076483718289490514920786900514455948318,1.77384602348831348224808584967077988220662147693458533736994544328647927674024709638273312775076823,3.15940316460338204905212219667436725212711344838416621469101803643908257264441070760944460956018308,4.08266180022139137082223684789042684510482079837381903743632015974891188580095989087613218488741040,3.54521498527191555344929234912436358603352085464143280984787285594175566002969862133007720637092265,0.79504803985560661765633440744338870827909649418056530515534702358422290128514719281600708684308901,-4.1189699882362159287269586840981260221877185536542825960411807458481243908245908165030759970204155]
        res_map_ddpsi = self.map_derivative2_segment(k, point_1)
        if np.linalg.norm(ref_map_ddpsi - res_map_ddpsi)/np.linalg.norm(ref_map_ddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative2_segment(k,point_1)" )  

        ref_map_dddpsi = [0,0,0,0.14425244177310293012772351615326821938392186326070623591284748309541697971450037565740045078888054,0.98302750524020653862381759907532754214218965324277136917272302033995387562256715750821893011774301,3.46559532444907485126999831420231954540948457864890797628494964143747546935491958921578825631619767,8.3677845260858706422312657368116107723997138217932469767213057009224746082943831892283893614897412,15.2440634911558828265982368890588793329310040786963741711822899046460595482266035107686218476469941,21.4703987192085515572517502404742357043880427334691035370040377276059826099334523628405013247623391,22.2014856060785099673067756193839351746298924717460194785220019545868060571206602617626928415780630]
        res_map_dddpsi = self.map_derivative3_segment(k, point_1)
        if np.linalg.norm(ref_map_dddpsi - res_map_dddpsi)/np.linalg.norm(ref_map_dddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative3_segment(k,point_1)" )          

        ref_map_ddddpsi = [0,0,0,0,0.2098217334881497165494160234956628645584318011064817976914145208660610614029096373198552011474626,1.7873227368003755247705774528642318948039811877141297621322236733453706829501221045603980547595327,7.8760314899392969775950294480485551231856140550755219175224664535263107979696435597233208155019888,23.802066988830014642176575764075842982470299754637137607730254926849817959061328155329548378637233,54.847741839645267223260970820114813375198285150853784388014798630338769446679341020048903432006029,100.869417001749694051986000795363842243770733046016762525848217469023890852601317958615995108119422]
        res_map_ddddpsi = self.map_derivative4_segment(k, point_1)
        if np.linalg.norm(ref_map_ddddpsi - res_map_ddddpsi)/np.linalg.norm(ref_map_ddddpsi)>1.0e-12:
            raise ValueError("test_basis_functions: map_derivative4_segment(k,point_1)" )      

        self._calc_domain = calc_domain_back
        return(self)

    def test_mappings(self):
        j = np.arange(0, self.__N)
        t = self.circlemap_in_t(j)
        t = np.append(t,self.__mp.pi)
        t = np.append(t, 0)
        x = self.from_t_map_to_infinity(t)
        t1 = self.from_infinity_map_to_t(x)
        rel_norm_mapping = np.linalg.norm(t-t1)/np.linalg.norm(t1)
        if rel_norm_mapping>1.0e-12:
            raise ValueError("test_mappings: error in t<->infinity mappings, error = " + str(rel_norm_mapping) )
    
    def test_mappings_segment(self):
        calc_domain_back = self._calc_domain.copy()
        self._calc_domain = [-4, 7]        
        j = np.arange(0, self.__N)
        t = self.circlemap_in_t(j)
        t = np.append(t,self.__mp.pi)
        t = np.append(t, 0)
        x = self.from_t_map_to_segment(t)
        t1 = self.from_segment_map_to_t(x)
        rel_norm_mapping = np.linalg.norm(t-t1)/np.linalg.norm(t1)
        if rel_norm_mapping>1.0e-12:
            raise ValueError("test_mappings: error in t<->segment mappings, error = " + str(rel_norm_mapping) )        
        self._calc_domain = calc_domain_back
        

    def test(self):
        self.test_basis_functions()
        self.test_mappings()
        self.test_basis_functions_segment()
        self.test_mappings_segment()

class basic_discretization(object):
    def __init__(self, N, domain, L = 1, use_mpmath = False, prec = 100):
        self._N = N
        self._M = self._N+2 # due to additional boundary constraints        
        self._additional_boundaries = 0
        self._domain = domain
        if len(self._domain) != 2:
            raise ValueError("domain should be provided as a list of format [x_min, x_max]")
        self._domain_type = "segment"
        if (np.max(self._domain) == np.inf) and (np.min(self._domain) == -np.inf):
            self._domain_type = "R"
        elif np.max(self._domain) == np.inf:
            self._domain_type = "R+"
        elif np.min(self._domain) == -np.inf:
            self._domain_type = "R-"

        #check for implemented
        if self._domain_type=="R" or self._domain_type=="R-":
            raise ValueError("basic_discretization: only segment and R+ domains are implemented")

        self._basis = basis_functions(N = N, L = L, use_mpmath = use_mpmath, prec = prec, calc_domain = self._domain)
        self._basis.test()
        #Dirichlet, 1st derivative, ..., 4th derivative
        self._boundary_conditions_left = [None, None, None, None, None]
        self._boundary_conditions_right = [None, None, None, None, None]
        self._boundaries = []

    def set_boundary_conditions(self, values):
        self._boundary_conditions_left = values[0]
        self._boundary_conditions_right = values[1]
        self._boundaries = []
        if self._domain_type == "R+":
            for idx, el in enumerate([self._boundary_conditions_right, self._boundary_conditions_left]):
                
                der = 0
                for j in el:
                    side = 'l' if idx == 0 else 'r'
                    self._boundaries.append([j, der, side])
                    der = der+1
        elif self._domain_type == "segment":
            for idx, el in enumerate([self._boundary_conditions_right, self._boundary_conditions_left]):
                der = 0
                for j in el:
                    side = 'l' if idx == 0 else 'r'
                    self._boundaries.append([j, der, side])    
                    der = der+1      
        else:
            raise ValueError("set_boundary_conditions: only segment and R+ domains are implemented")
        
        self._additional_boundaries = 0
        for bnd in self._boundaries:
            if bnd[0] != None:
                self._additional_boundaries = self._additional_boundaries + 1 

        # self._additional_boundaries = max(self._additional_boundaries, 0)
        # print(self._boundaries)
        # print(self._additional_boundaries)
        #set number of BCs!
    def is_using_mpmath(self):
        return self._basis.is_using_mpmath()

    def get_mpmath_ref(self):
        return self._basis.get_mpmath_ref()

    def psi(self,k,x):
        return self._basis.psi(k,x);
    def dpsi(self,k,x):
        if self._domain_type == "segment":
            return self._basis.map_derivative1_segment(k, x);
        elif self._domain_type == "R+":
            return self._basis.map_derivative1_inf(k, x);
    def ddpsi(self,k,x):
        if self._domain_type == "segment":
            return self._basis.map_derivative2_segment(k, x);
        elif self._domain_type == "R+":
            return self._basis.map_derivative2_inf(k, x);        
    def dddpsi(self,k,x):
        if self._domain_type == "segment":
            return self._basis.map_derivative3_segment(k, x);
        elif self._domain_type == "R+":
            return self._basis.map_derivative3_inf(k, x);          
    def ddddpsi(self, k, x):
        if self._domain_type == "segment":
            return self._basis.map_derivative4_segment(k, x);
        elif self._domain_type == "R+":
            return self._basis.map_derivative4_inf(k, x);                

    def psi_with_derivative(self, k, x, der = 0):
        if der == 0:
            return self.psi(k, x)
        elif der == 1:
            return self.dpsi(k, x)
        elif der == 2:
            return self.ddpsi(k, x)
        elif der == 3:
            return self.dddpsi(k, x)
        elif der == 4:
            return self.ddddpsi(k, x)
        else:
            raise ValueError("psi_with_derivative: implemented only up to 4-th derivative")

    def psi_with_derivative_value_at_left(self, k, der):
        if der>=1 and der<=4:
            if self._domain_type == "R+":
                raise ValueError("psi_with_derivative_value_at_right: on R+ domain regularity at inifinity is assumed.")
                if der==1:
                    return 0
                elif der==2:
                    return 0
                elif der==3:
                    return 0
                elif der==4:
                    return 0                
            elif self._domain_type == "segment":
                if der==1:
                    return self._basis.map_derivative1_segment_at_0(k)
                elif der==2:
                    return self._basis.map_derivative2_segment_at_0(k)
                elif der==3:
                    return self._basis.map_derivative3_segment_at_0(k)
                elif der==4:
                    return self._basis.map_derivative4_segment_at_0(k)
        else:
            raise ValueError("psi_with_derivative_value_at_left: only derivatives 1, 2, 3 and 4 are supported.")            

    # to be inserted at need
    def psi_with_derivative_value_at_right(self, k, der):
        if der>=1 and der<=4:
            if self._domain_type == "R+":     
                if der==1:
                    return self._basis.map_value_derivative1_inf_at_pi(k)
                elif der==2:
                    return self._basis.map_value_derivative2_inf_at_pi(k)
                elif der==3:
                    return self._basis.map_value_derivative3_inf_at_pi(k)
                elif der==4:
                    return self._basis.map_value_derivative4_inf_at_pi(k)
                else:
                    raise ValueError("psi_with_derivative_value_at_right: only derivatives 1, 2, 3 and 4 are supported.")
            elif self._domain_type == "segment":
                if der==1:
                    return self._basis.map_derivative1_segment_at_pi(k)
                elif der==2:
                    return self._basis.map_derivative2_segment_at_pi(k)
                elif der==3:
                    return self._basis.map_derivative3_segment_at_pi(k)
                elif der==4:
                    return self._basis.map_derivative4_segment_at_pi(k)
        else:
            raise ValueError("psi_with_derivative_value_at_left: only derivatives 1, 2, 3 and 4 are supported.")        



    def set_mapping_parameter(self, value):
        self._basis.set_mapping_parameter(value)

    def discrete_points_in_basis(self, j):
        return self._basis.circlemap_in_t(j)
    
    def discrete_points_in_domain(self, j):
        if self._domain_type == "R+":  
            pts = self._basis.from_t_map_to_infinity(self._basis.circlemap_in_t(j))
        elif self._domain_type == "segment":
            pts = self._basis.from_t_map_to_segment(self._basis.circlemap_in_t(j))
        else:
            raise ValueError("discrete_points_in_domain: only semgent and R+ domains are supported")
        return pts
    
    def discrete_points_in_basis_with_bounds(self, j):
        return self._basis.circlemap_with_bound_in_t(j)
    
    def discrete_points_in_domain_with_bounds(self, j):
        if self._domain_type == "R+":  
            pts = self._basis.from_t_map_to_infinity(self._basis.circlemap_with_bound_in_t(j))
        elif self._domain_type == "segment":
            pts = self._basis.from_t_map_to_segment(self._basis.circlemap_with_bound_in_t(j))
        else:
            raise ValueError("discrete_points_in_domain_with_bounds: only semgent and R+ domains are supported")
        return pts
    
    def all_discrete(self):
        return np.arange(0,self._N)

    def all_discrete_with_bounds(self):
        return np.arange(0,self._M)        

    def all_discrete_points_in_domain(self):
        k = np.arange(0,self._N)
        return self.discrete_points_in_domain(k)
    
    def all_discrete_points_in_basis(self):
        k = np.arange(0,self._N)
        return self.discrete_points_in_basis(k)        

    def all_discrete_points_in_domain_with_bounds(self):
        k = np.arange(0,self._M)
        return self.discrete_points_in_domain_with_bounds(k)
    
    def all_discrete_points_in_basis_with_bounds(self):
        k = np.arange(0,self._M)
        return self.discrete_points_in_basis_with_bounds(k)      

    def arg_from_basis_to_domain(self, t):
        if self._domain_type == "R+":  
            pts = self._basis.from_t_map_to_infinity(t)
        elif self._domain_type == "segment":
            pts = self._basis.from_t_map_to_segment(t)
        else:
            raise ValueError("arg_from_basis_to_domain: only semgent and R+ domains are supported")
        return pts

    
    def arg_from_domain_to_basis(self, x):
        if self._domain_type == "R+":  
            pts = self._basis.from_infinity_map_to_t(x)
        elif self._domain_type == "segment":
            pts = self._basis.from_segment_map_to_t(x)
        else:
            raise ValueError("arg_from_domain_to_basis: only semgent and R+ domains are supported")
        return pts       
    
    def basis_boundaries(self):
        return self._basis._domain


    def solution_in_basis(self, coeffs, t):
        M_ = len(coeffs)
        func = self.psi(0, t)*coeffs[0]
        for k in range(1, M_):
            func = func+self.psi(k, t)*coeffs[k]
        return func

    def solution_in_domain(self, coeffs, x):
        t = self.arg_from_domain_to_basis(x)
        return self.solution_in_basis(coeffs, t)

    def solution_with_derivative_in_basis(self, coeffs, t, der = 0):
        M_ = len(coeffs)
        func = self.psi_with_derivative(0, t, der)*coeffs[0]
        for k in range(1, M_):
            func = func+self.psi_with_derivative(k, t, der)*coeffs[k]
        return func        

    def solution_with_derivative_in_domain(self, coeffs, x, der = 0):
        t = self.arg_from_domain_to_basis(x)
        return self.solution_with_derivative_in_basis(coeffs, t, der)

    def print_matrix(self, M):
        print('\n'.join([' '.join(['{:4}'.format(item) for item in row]) for row in M]))


class collocation_discretization(basic_discretization):
    def __init__(self, N, domain, L = 1, use_mpmath = False, prec = 100):
        super().__init__(N, domain, L, use_mpmath, prec)
        

    def mass_matrix(self):
        if( super().is_using_mpmath() ):
            M = super().get_mpmath_ref().zeros(self._N+self._additional_boundaries, self._N+self._additional_boundaries)
        else:
            M = np.zeros((self._N+self._additional_boundaries, self._N+self._additional_boundaries))

        is_there_boundaries = 0 if self._additional_boundaries == 0 else 1

        for j in range(0, self._N+self._additional_boundaries):
            for k in range(0, self._N+self._additional_boundaries):
                if j>=is_there_boundaries and j<self._N+is_there_boundaries:
                    ppp = self.psi(k, self.discrete_points_in_basis(j-is_there_boundaries) ) 
                    M[j,k] = ppp[0]

        return(M)

    def bilinear_form(self, operator, alpha):
        if( super().is_using_mpmath() ):
            S = super().get_mpmath_ref().zeros(self._N+self._additional_boundaries, self._N+self._additional_boundaries)
        else:        
            S = np.zeros( (self._N+self._additional_boundaries, self._N+self._additional_boundaries) )
        is_there_boundaries = 0 if self._additional_boundaries==0 else 1

        for j in range(0, self._N+self._additional_boundaries):
            for k in range(0, self._N+self._additional_boundaries):
                 if j>=is_there_boundaries and j<self._N+is_there_boundaries:
                    ppp = alpha*operator(k, self.discrete_points_in_basis(j-is_there_boundaries) )
                    S[j,k] = ppp[0]#operator_in_basis_funcs(k, self.discrete_points_in_basis_with_bounds(j) )
        return(S)

    def set_boundary_bilinear_form_tau_method_old(self, S):
        matrix_boundary_rows = [0]
        for j in range(1,self._additional_boundaries):
            matrix_boundary_rows.append(self._N+j)

        # print("matrix_boundary_rows = ", matrix_boundary_rows)
        matrix_boundary_row_index = 0
        for b_value, index in zip(self._boundaries, range(len(self._boundaries)) ):
            if b_value[0] != None:
                if b_value[1] == 0:
                    position = 0 if index == 0 else 1 #check if Dirichlet is at positon 0 in basis or in position 1
                    for k in range(self._N+self._additional_boundaries):
                        ppp = self.psi(k, self.basis_boundaries()[position] )
                        if hasattr(ppp, "__len__"):
                            S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp[0]
                        else:
                            S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp

                    matrix_boundary_row_index = matrix_boundary_row_index + 1
                else:
                    der_id_val = b_value[1]
                    for k in range(self._N+self._additional_boundaries):
                        ppp = self.psi_with_derivative_value_at_left(k, der_id_val)
                        if hasattr(ppp, "__len__"):
                            S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp[0]
                        else:
                            S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp
                    matrix_boundary_row_index = matrix_boundary_row_index + 1

        return(S)

    def set_boundary_bilinear_form_tau_method(self, S):
        matrix_boundary_rows = [0]
        for j in range(1,self._additional_boundaries):
            matrix_boundary_rows.append(self._N+j)

        # print("matrix_boundary_rows = ", matrix_boundary_rows)
        matrix_boundary_row_index = 0
        for b in self._boundaries:
            #if Dirichlet boundary
            if b[0] != None and b[1] == 0:
                if b[2] == 'l':
                    position = 0
                elif b[2] == 'r':
                    position = 1
                for k in range(self._N+self._additional_boundaries):
                    ppp = self.psi(k, self.basis_boundaries()[position] )   
                    if hasattr(ppp, "__len__"):
                        S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp[0]
                    else:
                        S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp
                matrix_boundary_row_index = matrix_boundary_row_index + 1
            #if Neumann conditons
            elif b[0] != None and b[1] > 0:
                for k in range(self._N+self._additional_boundaries):
                    if b[2] == 'l':
                        ppp = self.psi_with_derivative_value_at_left(k, b[1])
                    elif b[2] == 'r':
                        ppp = self.psi_with_derivative_value_at_right(k, b[1])
                    if hasattr(ppp, "__len__"):
                        S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp[0]
                    else:
                        S[matrix_boundary_rows[matrix_boundary_row_index],k] = ppp
                matrix_boundary_row_index = matrix_boundary_row_index + 1
        return(S)

    def linear_functional(self, operator, **kwargs):
        x = self.all_discrete_points_in_domain()
        t = self.all_discrete_points_in_basis()
        if self._additional_boundaries>0:
            x = np.insert(x,[0],[0])
            t = np.insert(t,[0],[0])
        for j in range(self._additional_boundaries-1):
            x = np.append(x,0)
            t = np.append(t,0)
        return operator(x, t, **kwargs)


    def linear_functional_boundary(self, rhs):
        matrix_boundary_rows = [0]
        for j in range(1,self._additional_boundaries):
            matrix_boundary_rows.append(self._N+j)
        # print(matrix_boundary_rows)
        matrix_boundary_row_index = 0
        for b_value, index in zip(self._boundaries, range(len(self._boundaries)) ):
            if b_value[0] != None:
                rhs[ matrix_boundary_rows[matrix_boundary_row_index] ] = b_value[0]
                matrix_boundary_row_index = matrix_boundary_row_index + 1

        return rhs


    def linear_functional_linearization(self, operator, **kwargs):
        if( super().is_using_mpmath() ):
            N = super().get_mpmath_ref().zeros(self._N+self._additional_boundaries, self._N+self._additional_boundaries)
        else:
            N = np.zeros( (self._N+self._additional_boundaries, self._N+self._additional_boundaries) )

        is_there_boundaries = 0 if self._additional_boundaries==0 else 1
        x = self.all_discrete_points_in_domain()
        t = self.all_discrete_points_in_basis()
        for j in range(0, self._N+self._additional_boundaries):
            for k in range(0, self._N+self._additional_boundaries):
                if j>=is_there_boundaries and j<self._N+is_there_boundaries:
                    ppp = self.psi(k, t[j-is_there_boundaries])*operator(x[j-is_there_boundaries], t[j-is_there_boundaries], **kwargs)
                    if hasattr(ppp, "__len__"):
                        N[j, k] = ppp[0]
                    else:
                        N[j, k] = ppp
        return N

    def linear_functional_linearization_boundary(self, N):
        matrix_boundary_rows = [0]
        for j in range(1,self._additional_boundaries):
            matrix_boundary_rows.append(self._N+j)

        matrix_boundary_row_index = 0
        for b_value in self._boundaries:
            if b_value[0] != None:
                for k in range(self._N+self._additional_boundaries):
                    N[matrix_boundary_rows[matrix_boundary_row_index],k] = 0
                matrix_boundary_row_index = matrix_boundary_row_index + 1
        return N

    def solve(self, matrix, rhs):
        if( super().is_using_mpmath() ):
            sol = super().get_mpmath_ref().lu_solve(matrix, rhs)
        else:
            sol = np.linalg.solve(matrix, rhs)

        return sol

    def function_values(self, f):
        def func_wrapper(x, t):
            return f(x)

        vals = self.linear_functional(func_wrapper)
        vals = self.linear_functional_boundary(vals)
        
        return(vals)

    def function_expand(self, f):
        M = self.mass_matrix()
        M = self.set_boundary_bilinear_form_tau_method(M)
        b = self.function_values(f)
        if( super().is_using_mpmath() ):
            c = super().get_mpmath_ref().lu_solve(M, b)
        else:
            c = np.linalg.solve(M, b)
        
        return c

    def L2_error(self, c, function):
        def func_in_basis(t):
            return function(self.arg_from_basis_to_domain(t))

        def difference(t):
            return np.power((func_in_basis(t) - self.solution_in_basis(c, t)),2)
        
        ref = integrate.quad( lambda t: np.power(func_in_basis(t),2), self.basis_boundaries()[0], self.basis_boundaries()[1] )

        res = integrate.quad( difference, self.basis_boundaries()[0], self.basis_boundaries()[1] )
        return np.sqrt(res[0]/ref[0])

    
# class galerkin_discretization(basic_discretization):
#     def __init__(self, N, domain):
#         super().__init__(N, domain)

#     def mass_matrix(self):
#         M = np.zeros((self._M, self._M))
#         for j in range(0, self._M):
#             for k in range(0, self._M):
#                     if j==0:
#                         M[j,k] = self.psi(k, self.basis_boundaries()[0] ) #boundary 0
#                     elif j==self._M-1:
#                         M[j,k] = self.psi(k, self.basis_boundaries()[1] ) #boundary 1
#                     else:
#                         res = integrate.quad( lambda t: self.psi(j, t)*self.psi(k, t), self.basis_boundaries()[0], self.basis_boundaries()[1] )
#                         M[j,k] = res[0]
#         return(M)

#     def bilinear_form(self, operator_cols, operator_rows, alpha):
#         S = np.zeros((self._M, self._M))
#         x_all = self.all_discrete_points_in_basis_with_bounds()
#         x0 = x_all[0]
#         for j in range(0, self._M):
#             for k in range(0, self._M):
#                 def psipsi(t):
#                     return alpha*operator_cols(k, t)*operator_rows(j, t)

#                 res = integrate.quad(psipsi, 0, np.pi)
#                 S[j,k] = S[j,k] + res[0]

                    
#         return(S)

#     def set_boundary_bilinear_form_tau_method(self, S):
#         for j in range(0, self._M):
#             for k in range(0, self._M):
#                     if j==0:
#                         S[j,k] = self.psi(k, self.basis_boundaries()[0] ) #boundary 0
#                     elif j==self._M-1:
#                         S[j,k] = self.psi(k, self.basis_boundaries()[1] ) #boundary 1
#         return(S)

#     def linear_functional(self, operator, **kwargs):
#         x = self.all_discrete_points_in_domain_with_bounds()
#         t = self.all_discrete_points_in_basis_with_bounds()
#         return operator(x, t, **kwargs)

#     def linear_functional_linearization(self, operator, **kwargs):
#         N = np.zeros((self._M, self._M))
#         x = self.all_discrete_points_in_domain_with_bounds()
#         t = self.all_discrete_points_in_basis_with_bounds()
#         for j in range(0, self._M):
#             for k in range(0, self._M):
#                 N[j, k] = self.psi(k, t[j])*operator(x[j], t[j], **kwargs)
#         return N
        
#     def linear_functional_linearization_boundary(self, N):
#         for j in range(0, self._M):
#             for k in range(0, self._M):
#                     if j==0:
#                         N[j,k] = 0*self.psi(k, self.basis_boundaries()[0] ) #boundary 0
#                     elif j==self._M-1:
#                         N[j,k] = 0*self.psi(k, self.basis_boundaries()[1] ) #boundary 1
#         return N

#     def solve(self, matrix, rhs):
#         idiag = np.diag(1/np.diag(matrix))
#         PA = np.dot(idiag,matrix)
#         Pb = np.dot(idiag, rhs)
#         sol = np.linalg.solve(PA, Pb)
#         return sol

#     def function_values(self, f):
#         k = np.arange(0, self._M)
#         x = self.discrete_points_in_domain_with_bounds(k)
#         return(f(x), x)

#     def function_expand(self, f):
#         b = np.zeros(self._M)
#         for j in range(0,self._M):
#             def fpsi(t):
#                 return f(self.arg_from_basis_to_domain(t))*self.psi(j, t)
#             res = integrate.quad(fpsi, self.basis_boundaries()[0], self.basis_boundaries()[1] )
#             b[j] = res[0]
#         return b

#     def get_dirichelt(self):
#         if self._domain_type == "R+":
#             return [self._dirichelt[1], self._dirichelt[0]]
#         else:
#             return [self._dirichelt[0], self._dirichelt[1]]


#     def rhs_expand(self, rhs):
#         b = self.function_expand(rhs)
#         if self._dirichelt[0] != None: 
#             if self._domain_type == "R+":
#                 b[0] = self._dirichelt[1]
#             else:
#                 b[0] = self._dirichelt[0]
#         if self._dirichelt[1] != None:
#             if self._domain_type == "R+":
#                 b[-1:] = self._dirichelt[0]
#             else:
#                 b[-1:] = self._dirichelt[1]
#         return b

#     def linear_functional_boundary(self, rhs):
#         if self._dirichelt[0] != None: 
#             if self._domain_type == "R+":
#                 rhs[0] = self._dirichelt[1]
#             else:
#                 rhs[0] = self._dirichelt[0]
#         if self._dirichelt[1] != None:
#             if self._domain_type == "R+":
#                 rhs[-1:] = self._dirichelt[0]
#             else:
#                 rhs[-1:] = self._dirichelt[1]
#         return rhs
    

class solve_nonlinear_problem(collocation_discretization):
    def __init__(self, N, domain, L = 1, tolerance = 1.0e-8, use_method = "newton", visualize = False, total_iterations = 50, use_globalization = True, globalization_init = 1.0, use_mpmath = False, prec = 100, use_adjoint_opimization = False):  #log_10(2^53) \sim 16
        super().__init__(N = N, domain = domain, L = L, use_mpmath = use_mpmath, prec = prec)
        self._N = N
        self.__use_method = use_method # "root", "nonlin_solve", "newton"
        self.__visualize = visualize #can be used only in "newton" method
        self.__tolerance = tolerance
        self.__total_iterations = total_iterations
        self.__problem = None
        self.__S = None
        self.__iS = None
        self.__use_globalization = use_globalization
        self.__globalization_init = globalization_init
        self.__globalization = 0
        self.__base_solution = None
        self.__converged = False
        self.__use_adjoint_opimization = use_adjoint_opimization
        self.__use_mpmath = super().is_using_mpmath()
        self.__mp_ref = super().get_mpmath_ref()
        if self.__use_mpmath:
            self.__dot = self.__mp_ref.fdot
            self.__sqrt = self.__mp_ref.sqrt
            self.__norm = self.__mp_ref.norm
            self.__inv = self.__mp_ref.inverse
        else:
            self.__dot = np.dot    
            self.__sqrt = np.sqrt
            self.__norm = np.linalg.norm
            self.__inv = np.linalg.inv



    def __form_matrices(self):
        alpha = self.__problem.left_hand_side()
        self.__S  = self.bilinear_form(self.psi, 0)
        if alpha[0] != None:
            self.__S  = self.__S + self.bilinear_form(self.psi, alpha[0])
        if alpha[1] != None:
            self.__S = self.__S + self.bilinear_form(self.dpsi, alpha[1])
        if alpha[2] != None:
            self.__S = self.__S + self.bilinear_form(self.ddpsi, alpha[2])
        if alpha[3] != None:
            self.__S = self.__S + self.bilinear_form(self.dddpsi, alpha[3])
        if alpha[4] != None:
            self.__S = self.__S + self.bilinear_form(self.ddddpsi, alpha[4])

        self.__S = self.set_boundary_bilinear_form_tau_method(self.__S )
        # print(self.__S)
        self.__iS = self.__inv(self.__S)

        # print("alpha = ", alpha)
        # print("S = ", self.__S )

    def __right_hand_side(self, c, q):
        # x = self.all_discrete_points_in_domain_with_bounds()
        # t = self.all_discrete_points_in_basis_with_bounds()
        # u = self.solution_in_basis(c, t)
        # rhs = self.__problem.right_hand_side(x, u, q)
        # rhs = self.rhs_boundary(rhs)
        def local_operator(x, t, c, q):
            u = self.solution_in_basis(c, t) 
            return self.__problem.right_hand_side(x, u, q)

        rhs = self.linear_functional(local_operator, c=c, q=q)
        rhs = self.linear_functional_boundary(rhs)
        return rhs
    
    def __right_hand_side_linearization(self, c, q):
        # x = self.all_discrete_points_in_domain_with_bounds()
        # t = self.all_discrete_points_in_basis_with_bounds()
        # u = self.solution_in_basis(c, t)
        # f = self.__problem.right_hand_side_linearization(x, u, q)
        def local_operator(x, t, c, q):
            u = self.solution_in_basis(c, t) 
            return self.__problem.right_hand_side_linearization(x, u, q)
        
        N = self.linear_functional_linearization(local_operator, c=c, q=q)
        N = self.linear_functional_linearization_boundary(N)
        return N

    def __residual_operator(self, c):
        rhs = self.__right_hand_side(c, self.__globalization)
        if self.__use_mpmath:
            lhs = self.__S*c#self.__dot(self.__S, c)
        else:
            lhs = self.__dot(self.__S, c)
        res = rhs-lhs
        return res
        
    def __linearization_operator(self, c):
        N = self.__right_hand_side_linearization(c, self.__globalization)
        A = -(self.__S-N)
        return A

    def __base_solution_expand(self):
        c0 = self.function_expand(self.__problem.base_solution)
        return c0
    
    def __base_solution_values(self):
        c0 = self.function_values(self.__problem.base_solution)
        return c0

    def __set_boundary_conditoins_on_domain(self, values):
        self.set_boundary_conditions(values)

    def set_method(self, method_name):
        self.__use_method = method_name

    def set_problem(self, problem):
        self.__problem = problem
        if self.__use_mpmath:
            self.__problem.set_use_mp_math(self.__mp_ref)
        self.__set_boundary_conditoins_on_domain(self.__problem.get_boundary_conditions() )
        self.__form_matrices()

    def residual_L2_norm(self, c):
        relnorm = self.__norm(self.__residual_operator(c))/self.__sqrt(self._M)
        return relnorm

    def solve_problem(self, c0 = None):
        if self.__problem==None:
            raise ValueError("one should call 'set_problem' before attempring to solve one.")
        if self.__use_mpmath:
            print("Warning! Using mpmath with", self.__mp_ref.mp.dps, "significant digits and backend", self.__mp_ref.libmp.BACKEND ,". Solution will be slow!" )
        if not self.__use_mpmath and self.__use_adjoint_opimization:
            print("Using initial guess update by the adjoint optimization.")

        converged = False
        c_ref = self.__base_solution_expand()
        if np.any(c0) != None:
            self.__base_solution = c0   
        if np.any(self.__base_solution) == None:
            c0 = self.__base_solution_expand()
        else:
            c0 = self.function_expand( lambda x: self.solution_in_domain(self.__base_solution, x) )
        c = c0;
        self.__globalization = 0

        # def fun(x):
        #     vec = self.__residual_operator(x);
        #     # iSvec = np.dot(self.__iS,vec)
        #     iSvec = vec
        #     return np.dot(iSvec, iSvec)
        # def jac(x):
        #     A = self.__linearization_operator(x)
        #     AT = A.transpose()
        #     valT = -2*np.dot(AT, x)
        #     # iSval = np.dot(self.__iS,valT)
        #     return valT

        # if not self.__use_mpmath and self.__use_adjoint_opimization:
        #     res = minimize(fun, c, method = self.__adjoint_method, jac = jac)
        #     print(res)
        #     c = res.x

        # c = self.adjoint_descent(c, 1)    
        
        if not self.__use_mpmath and self.__use_adjoint_opimization:
            res = optimize.fsolve(self.__residual_operator, c, fprime=self.__linearization_operator, xtol = 1.0e-1, full_output = True)
            # print(res)
            c = res[0]

        if self.__use_method == "root":
            if self.__use_mpmath:
                raise ValueError("only 'newton' can be used to solve with mpmath library")            
            res = optimize.root(self.__residual_operator, c, tol=self.__tolerance*self.__sqrt(self._M), jac=self.__linearization_operator )
            converged = res.success
            c = res.x;
        elif self.__use_method == "root_no_jac":
            if self.__use_mpmath:
                raise ValueError("only 'newton' can be used to solve with mpmath library")            
            res = optimize.root(self.__residual_operator, c, tol=self.__tolerance*self.__sqrt(self._M))
            converged = res.success
            c = res.x;
        elif self.__use_method == "nonlin_solve":
            if self.__use_mpmath:
                raise ValueError("only 'newton' can be used to solve with mpmath library")                        
            try:
                c = optimize.nonlin.nonlin_solve(self.__residual_operator, c, jac=self.__linearization_operator)
            except:
                converged = False

        elif self.__use_method == "newton":
            residual_norm_0 = self.__sqrt(self._M)#self.residual_L2_norm(c_ref)
            residual_norm = residual_norm_0
            iterations = 0
            print("newton: relative target tolerance = {:.6e}".format(self.__tolerance*np.float64(residual_norm_0)))
            for j_homotopy in range(0,100):
                c_save = c.copy()
                while residual_norm>self.__tolerance*residual_norm_0 and iterations<self.__total_iterations: 
                    iterations = iterations + 1
                    A = -self.__linearization_operator(c)
                    b = self.__residual_operator(c)
                    dc = self.solve(A, b)# np.linalg.solve(A, b)
                    w = 1
                    converged = True
                    residual_norm_new = residual_norm*(1+1/10)
                    while residual_norm_new>=residual_norm:
                        c_new = c + w*dc
                        residual_norm_new = self.residual_L2_norm(c_new)
                        w = w/2
                        if w<1.0e-10 or not self.__mp_ref.isfinite(residual_norm_new):
                            converged = False
                            break
                    c = c_new
                    if not converged:
                        print("newton: failed to converge for old norm: {:.6e}".format(np.float64(residual_norm)), "with new norm: {:.6e}".format(np.float64(residual_norm_new)), " with weight: {:.6e}".format(np.float64(w)) )
                        break
                    residual_norm = residual_norm_new
                    print("newton: iteration ", iterations, ", residual norm {:.6e}".format(np.float64(residual_norm)), ", weight = {:.6e}".format(np.float64(w*2)) )
                    if self.__visualize:
                        t = np.arange(0, np.pi, 0.01)
                        u = self.obtain_solution_in_basis(c, t)
                        plt.plot(t, u, label = iterations)

                if self.__visualize:
                    plt.legend()
                    plt.show()

                if iterations>=self.__total_iterations:
                    converged = False

                if not converged or not self.__mp_ref.isfinite(residual_norm):
                    iterations = 0
                    residual_norm = residual_norm_0
                    if self.__globalization == 0:
                        self.__globalization = self.__globalization_init
                    c = 0*c_save.copy() #reset data

                    self.__globalization = self.__globalization*1.25
                    if not self.__use_globalization:
                      break;
                else:
                    iterations = 0
                    residual_norm = residual_norm_0
                    if self.__globalization == 0.0:
                        break
                    elif self.__globalization < 1.0e-11:
                        self.__globalization = 0.0
                    else:
                        self.__globalization = self.__globalization*0.5
                        

                print("newton: residual norm = {:.6e}".format(np.float64(self.residual_L2_norm(c))), " with homotopy = {:.6e}".format(np.float64(self.__globalization)))
                if self.__globalization == 0.0:
                    break                

        elif self.__use_method == "print":
            k = np.arange(0, self._M)
            j = self.all_discrete_points_in_basis_with_bounds()
            print(j[self._M-2])
            print(self.dddpsi(k, j[self._M-1] ))
            print(sum(self.dddpsi(k, j[self._M-1] ) ))
            print(self.dddpsi(10, j[self._M-1] ))

        if converged:
            print("converged with norm = {:.6e}".format(np.float64(self.residual_L2_norm(c) ) ))
        else:
            print("failed to converge with norm = {:.6e}".format(np.float64(self.residual_L2_norm(c) ) ))
            
        self.__current_solution = c
        self.__converged = converged
        return c

    def is_converged(self):
        return self.__converged

    def obtain_solution_in_domain(self, c, x):
        return self.solution_in_domain(c, x)
    
    def obtain_solution_in_basis(self, c, t):
        return self.solution_in_basis(c, t)        
    
    def obtain_rhs_in_domain(self, c, x, problem):
        t = self.arg_from_domain_to_basis(x)
        u = self.solution_in_basis(c, t)
        return problem.right_hand_side(x, u, 0)
    
    def obtain_rhs_in_basis(self, c, t, problem):
        x = self.arg_from_basis_to_domain(t)
        u = self.solution_in_basis(c, t)  
        return problem.right_hand_side(x, u, 0) 
    
    def __save_data(self, file_name, data):
        if self.__use_mpmath:
            print("Warning! Writing data to file is performed in float128 binary format, some data may be lost")
        
        data_1 = np.float128(data)
        with open(file_name, 'wb') as handle:
            pickle.dump(data_1, handle)

    def __load_data(self, file_name):
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
        if self.__use_mpmath:
            print("Warning! Reading data from file is performed in float128 binary format, some data may be lost")
            # sizes = np.shape(data)
            # data_1 = self.__mp_ref.zeros(sizes[0],sizes[1])
            data_1 = self.__mp_ref.matrix(data.tolist())
        else:
            data_1 = data
        return data_1

    def save_solution(self, file_name, c):
        self.__save_data(file_name, c)

    def load_data(self, file_name):
        c = self.__load_data(file_name)
        return c

    def set_base_solution(self, file_name):
        self.__base_solution = self.__load_data(file_name)

    def reset_base_solution(self):
        self.__base_solution = None

    def use_adjoint_opimization(self, val):
        if type(val) != bool:
            raise ValueError("Only bool argument can be passed.")
        self.__use_adjoint_opimization = val


    #for a tests, but seem to fail in this problem
    def adjoint_descent(self, c, exec_time):
        def fun(t, x):
            print(t)
            vec = self.__residual_operator(x);
            A = self.__linearization_operator(x)
            AT = A.transpose()
            valT = -2*np.dot(AT, vec)
            valiST = np.dot(self.__iS, valT)
            return valiST

        r = ode(fun).set_integrator('vode', rtol = 1.0e-3, atol = 1.0e-5)
        r.set_initial_value(c, 0)

        residual_norm_b4 = self.residual_L2_norm(c)
        dt = exec_time/1000;
        while r.successful() and r.t < exec_time:
            r.integrate(r.t+dt)


        print(r)
        c = r.y
        residual_norm_aft = self.residual_L2_norm(c)
        print("residual_norm_b4 = ", residual_norm_b4 , " residual_norm_aft = ", residual_norm_aft )
        return c