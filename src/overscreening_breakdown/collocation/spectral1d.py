import numpy as np
from scipy import optimize, integrate
from matplotlib import pyplot as plt
import pickle
import mpmath #for custom pecision arithmetic
from inspect import isfunction, isclass
from numbers import Integral

from overscreening_breakdown.nonlinear_solver import LinearSolveError, NonlinearSystem
from overscreening_breakdown.continuation import solve_continuation
from overscreening_breakdown.collocation.adapter import CollocationProblemAdapter, prepare_collocation_problem
from overscreening_breakdown.arithmetic import NumpyArithmetic, MpmathArithmetic
from overscreening_breakdown.rational_mapping import RationalHalfLineMap
from overscreening_breakdown.initialization import initialize_fsolve, validate_initializer
from overscreening_breakdown.collocation.globalization import CollocationMeritGeometry
from overscreening_breakdown.scipy_solver import verify_scipy_result
from overscreening_breakdown.diagnostics import ProgressReporter


class basis_functions(object):
    def __init__(self, N, L = 1, use_mpmath = False, prec = 100, calc_domain = [0, np.inf]):
        self.__N = N
        self.__M = N+2
        self.__boundary_sides = ("l", "r")
        self.__L = L
        self._calc_domain = calc_domain
        self.__configure_backend(use_mpmath, prec)

    def __configure_backend(self, use_mpmath, prec=100):
        self.__use_mpmath = bool(use_mpmath)
        self.__mp = mpmath if self.__use_mpmath else np
        if self.__use_mpmath:
            self.__mp.mp.dps = prec
        self._domain = (0, self.__mp.pi)
        self.__cos = np.vectorize(self.__mp.cos)
        self.__sin = np.vectorize(self.__mp.sin)
        self.__acos = np.vectorize(
            self.__mp.acos if self.__use_mpmath else np.arccos
        )
        self.__sqrt = np.vectorize(self.__mp.sqrt)
        self.__tan = np.vectorize(self.__mp.tan)
        self.__half_line_key = None
        self.__half_line_mapping = None

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
    
    def set_domain(self, domain):
        self._calc_domain = domain

    def use_mpmath(self, prec = 100):
        self.__configure_backend(True, prec)

    def use_numpy(self):
        self.__configure_backend(False)

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
    
    def set_boundary_layout(self, sides):
        """Keep tau-row coordinates consistent with the actual boundary count."""
        sides = tuple(sides)
        if any(side not in ("l", "r") for side in sides):
            raise ValueError("boundary sides must be 'l' or 'r'")
        self.__boundary_sides = sides
        self.__M = self.__N + len(sides)

    def circlemap_with_bound_in_t(self, j):
        indices = np.asarray(j)
        if np.any(indices < 0) or np.any(indices >= self.__M):
            raise ValueError("boundary collocation index is outside the active rows")
        dtype = object if self.__use_mpmath else np.float64
        values = np.empty(indices.shape, dtype=dtype)
        for index in np.ndindex(indices.shape):
            row = int(indices[index])
            if self.__boundary_sides and (row == 0 or row > self.__N):
                boundary = 0 if row == 0 else row-self.__N
                side = self.__boundary_sides[boundary]
                values[index] = self.__mp.pi if side == "r" else 0
            else:
                interior = row-1 if self.__boundary_sides else row
                values[index] = self.__mp.pi*(2*interior+1)/(2*self.__N)
        return values[()] if values.ndim == 0 else values

    def __half_line_map(self):
        bits = self.__mp.mp.prec if self.__use_mpmath else 53
        key = (self.__use_mpmath, self.__L, bits)
        if key != self.__half_line_key:
            if self.__use_mpmath:
                arithmetic = MpmathArithmetic(self.__mp.mp.dps)
                # Legacy collocation follows global precision, including changes
                # that leave its rounded decimal-digit count unchanged.
                arithmetic.context.prec = bits
            else:
                arithmetic = NumpyArithmetic()
            self.__half_line_mapping = RationalHalfLineMap(arithmetic, L=self.__L)
            self.__half_line_key = key
        return self.__half_line_mapping

    def __map_half_line(self, method_name, values):
        mapping = self.__half_line_map()
        method = getattr(mapping, method_name)
        # Keep precision and complex values intact until scalar validation.
        inputs = np.asarray(values, dtype=object)
        dtype = object if self.__use_mpmath else np.float64
        output = np.empty(inputs.shape, dtype=dtype)
        for index in np.ndindex(inputs.shape):
            value = method(inputs[index])
            output[index] = self.__mp.mpf(value) if self.__use_mpmath else value
        return output[()] if inputs.ndim == 0 else output

    def from_infinity_map_to_t(self, x):
        """Map physical scalars or arrays to angles, with the wall at pi."""
        return self.__map_half_line("to_angle", x)

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
        """Map angles to physical coordinates, preserving array shape."""
        return self.__map_half_line("from_angle", t)

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
        self._M = self._N
        self._additional_boundaries = 0
        # self._domain = domain
        self._basis = basis_functions(N = N, L = L, use_mpmath = use_mpmath, prec = prec, calc_domain = domain)
        self._basis.test()
        self.set_domain(domain)
        self._basis.set_boundary_layout(())
        #Dirichlet, 1st derivative, ..., 4th derivative
        self._boundary_conditions_left = [None, None, None, None, None]
        self._boundary_conditions_right = [None, None, None, None, None]
        self._boundaries = []

    def set_domain(self, domain):
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
        self._basis.set_domain(domain)
        self._basis.test()

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

        self._M = self._N + self._additional_boundaries
        self._basis.set_boundary_layout(
            boundary[2] for boundary in self._boundaries if boundary[0] is not None
        )
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
    
    def set_domain(self, domain):
        super().set_domain(domain)

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
        if isfunction(alpha) or isclass(alpha): #isclass ok? operator () in python?
            call_alpha = True
        else:
            call_alpha = False

        if( super().is_using_mpmath() ):
            S = super().get_mpmath_ref().zeros(self._N+self._additional_boundaries, self._N+self._additional_boundaries)
        else:        
            S = np.zeros( (self._N+self._additional_boundaries, self._N+self._additional_boundaries) )
        is_there_boundaries = 0 if self._additional_boundaries==0 else 1

        for j in range(0, self._N+self._additional_boundaries):
            for k in range(0, self._N+self._additional_boundaries):
                 if j>=is_there_boundaries and j<self._N+is_there_boundaries:
                    if call_alpha:
                        all_x = self.discrete_points_in_domain(j - is_there_boundaries)
                        alpha_l = alpha(all_x)
                    else:
                        alpha_l = alpha

                    ppp = alpha_l*operator(k, self.discrete_points_in_basis(j-is_there_boundaries) )
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

    def linear_functional(self, operator, interior_only=False, **kwargs):
        x = self.all_discrete_points_in_domain()
        t = self.all_discrete_points_in_basis()
        if interior_only:
            # Boundary rows contain prescribed data, not source evaluations.
            # In particular, f need not be defined at the dummy x=0 formerly
            # inserted for these rows.
            dtype = object if self.is_using_mpmath() else np.float64
            values = np.asarray(operator(x, t, **kwargs), dtype=dtype)
            if values.shape != x.shape:
                raise ValueError("interior callback must preserve the collocation point shape")
            rhs = np.zeros(self._N+self._additional_boundaries, dtype=dtype)
            start = 1 if self._additional_boundaries else 0
            rhs[start:start+self._N] = values
            return rhs
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
        for point in range(self._N):
            row = point + is_there_boundaries
            coefficient = np.asarray(operator(x[point], t[point], **kwargs)).item()
            for column in range(self._N+self._additional_boundaries):
                value = self.psi(column, t[point])*coefficient
                N[row, column] = np.asarray(value).item()
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

    def function_values(self, f, interior_only=False):
        def func_wrapper(x, t):
            return f(x)

        vals = self.linear_functional(func_wrapper, interior_only=interior_only)
        vals = self.linear_functional_boundary(vals)
        
        return(vals)

    def function_expand(self, f, interior_only=False):
        M = self.mass_matrix()
        M = self.set_boundary_bilinear_form_tau_method(M)
        b = self.function_values(f, interior_only=interior_only)
        if( super().is_using_mpmath() ):
            c = super().get_mpmath_ref().lu_solve(M, b)
        else:
            c = np.linalg.solve(M, b)
        
        return c

    def L2_error(self, c, function):
        def func_in_basis(t):
            return function(self.arg_from_basis_to_domain(t))

        def reference(t):
            return np.asarray(np.power(func_in_basis(t), 2)).item()

        def difference(t):
            return np.asarray(np.power(func_in_basis(t) - self.solution_in_basis(c, t), 2)).item()
        
        ref = integrate.quad( reference, self.basis_boundaries()[0], self.basis_boundaries()[1] )

        res = integrate.quad( difference, self.basis_boundaries()[0], self.basis_boundaries()[1] )
        return np.sqrt(res[0]/ref[0])


class solve_nonlinear_problem(collocation_discretization):
    def __init__(self, N, domain = None, problem = None, L = 1, tolerance = 1.0e-8, use_method = "newton", visualize = False, total_iterations = 50, use_globalization = True, use_mpmath = False, prec = 100, *, initializer=None, initializer_tolerance="0.1", initializer_max_evaluations=200, newton_fallback=None, adjoint_max_steps=20, verbosity=0, progress_callback=None, progress_interval=1.0, continuation=None, _reporter=None):
        validate_initializer(initializer, use_mpmath=use_mpmath,
                             tolerance=initializer_tolerance,
                             max_evaluations=initializer_max_evaluations)
        if newton_fallback not in (None, "adjoint"):
            raise ValueError("newton_fallback must be None or 'adjoint'")
        if newton_fallback is not None and use_method != "newton":
            raise ValueError("adjoint recovery requires the newton method")
        if (isinstance(adjoint_max_steps, bool) or not isinstance(adjoint_max_steps, Integral)
                or adjoint_max_steps < 0):
            raise ValueError("adjoint_max_steps must be a nonnegative integer")
        self.__continuation = continuation
        self.__initializer = initializer
        self.__initializer_tolerance = initializer_tolerance
        self.__initializer_max_evaluations = int(initializer_max_evaluations)
        self.__newton_fallback = newton_fallback
        self.__adjoint_max_steps = int(adjoint_max_steps)
        self.__owns_reporter = _reporter is None
        self.__reporter_options = dict(verbosity=verbosity, progress_callback=progress_callback,
                                      progress_interval=progress_interval)
        self.__reporting_arithmetic = (MpmathArithmetic(prec) if use_mpmath
                                       else NumpyArithmetic())
        self.__progress = (_reporter if _reporter is not None else ProgressReporter(
            arithmetic=self.__reporting_arithmetic, solver="collocation", method=use_method,
            **self.__reporter_options)).bind(norm="rms")
        if problem is not None and domain is None:
            problem = prepare_collocation_problem(problem, use_mpmath=use_mpmath, prec=prec)
        if domain != None and problem == None:
            super().__init__(N = N, domain = domain, L = L, use_mpmath = use_mpmath, prec = prec)
            self.__common_constructor_operations(N = N, tolerance = tolerance, use_method = use_method, visualize = visualize, total_iterations = total_iterations, use_globalization = use_globalization, use_mpmath = use_mpmath, prec = prec)
        elif problem != None and domain == None:
            super().__init__(N = N, domain = problem.get_domain(), L = L, use_mpmath = use_mpmath, prec = prec)
            self.__common_constructor_operations(N = N, tolerance = tolerance, use_method = use_method, visualize = visualize, total_iterations = total_iterations, use_globalization = use_globalization, use_mpmath = use_mpmath, prec = prec)
            self.set_problem(problem)
        else:
            raise ValueError("solve_nonlinear_problem: either 'problem' or 'domain' must be provided in the constructor.")

    def __common_constructor_operations(self, N, tolerance, use_method, visualize, total_iterations, use_globalization, use_mpmath, prec):
        self._N = N
        self.__use_method = use_method # "root", "nonlin_solve", "newton"
        self.__visualize = visualize #can be used only in "newton" method
        self.__tolerance = tolerance
        self.__total_iterations = total_iterations
        self.__problem = None
        self.__S = None
        self.__use_globalization = use_globalization
        self.__base_solution = None
        self.__converged = False
        self.__newton_result = None
        self.__initialization_result = None
        self.__scipy_result = None
        self.__use_mpmath = super().is_using_mpmath()
        self.__precision = prec
        self.__mp_ref = super().get_mpmath_ref()
        if self.__use_mpmath:
            self.__dot = self.__mp_ref.fdot
            self.__sqrt = self.__mp_ref.sqrt
            self.__norm = self.__mp_ref.norm
        else:
            self.__dot = np.dot    
            self.__sqrt = np.sqrt
            self.__norm = np.linalg.norm


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

        rhs = self.linear_functional(
            local_operator, interior_only=isinstance(self.__problem, CollocationProblemAdapter),
            c=c, q=q,
        )
        rhs = self.linear_functional_boundary(rhs)
        if self.__use_mpmath:
            rhs = self.__mp_ref.matrix(list(rhs))
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

    def __residual_operator(self, c, q=0):
        if q != 0:
            raise ValueError("physical residual requires q=0; use the continuation wrapper")
        rhs = self.__right_hand_side(c, 0)
        if self.__use_mpmath:
            lhs = self.__S*c#self.__dot(self.__S, c)
        else:
            lhs = self.__dot(self.__S, c)
        res = rhs-lhs
        return res
        
    def __linearization_operator(self, c, q=0):
        if q != 0:
            raise ValueError("physical Jacobian requires q=0; use the continuation wrapper")
        N = self.__right_hand_side_linearization(c, 0)
        A = -(self.__S-N)
        return A

    def __base_solution_expand(self):
        c0 = self.function_expand(
            self.__problem.base_solution,
            interior_only=isinstance(self.__problem, CollocationProblemAdapter),
        )
        return c0
    
    def __base_solution_values(self):
        c0 = self.function_values(
            self.__problem.base_solution,
            interior_only=isinstance(self.__problem, CollocationProblemAdapter),
        )
        return c0

    def __set_boundary_conditoins_on_domain(self, values):
        self.set_boundary_conditions(values)

    def set_method(self, method_name):
        self.__use_method = method_name

    def set_tolerance(self, tolerance):
        self.__tolerance = tolerance

    def set_domain(self, domain):
        super().set_domain(domain)

    def set_problem(self, problem):
        problem = prepare_collocation_problem(
            problem, use_mpmath=self.__use_mpmath, prec=self.__precision,
        )
        validate = getattr(problem, "validate_collocation_compatibility", None)
        if validate is not None:
            validate()
        self.__problem = problem
        self.set_domain( problem.get_domain() )
        if self.__use_mpmath and not isinstance(problem, CollocationProblemAdapter):
            self.__problem.set_use_mp_math(self.__mp_ref)
        self.__set_boundary_conditoins_on_domain(self.__problem.get_boundary_conditions() )
        with self.__progress.phase("collocation_assembly", level=1,
                                   N=self._N, unknown_count=self._M):
            self.__form_matrices()

    def nonlinear_system(self, initial_guess=None):
        """Expose the physical callbacks and native merit geometry.

        Continuation wraps this fixed physical system instead of modifying
        the physical RHS callbacks or their shared problem description.
        """
        scalar = self.__mp_ref.mpf if self.__use_mpmath else np.float64
        initial = (self.__base_solution_expand() if initial_guess is None
                   else initial_guess.copy())
        if len(initial) != self._M:
            raise ValueError("initial coefficient count does not match the collocation basis")

        def solve_linear(matrix, rhs):
            try:
                return self.solve(matrix, rhs)
            except (np.linalg.LinAlgError, ZeroDivisionError) as error:
                raise LinearSolveError(str(error)) from error

        merit_options = {}
        if self.__newton_fallback == "adjoint":
            # Use the actual tau rows, including nonzero prescribed data.
            # The direction preserves their current traces; Newton still
            # checks the full residual, including every boundary equation.
            arithmetic = MpmathArithmetic(self.__precision) if self.__use_mpmath else NumpyArithmetic()
            indices = ([0] + [self._N+j for j in range(1, self._additional_boundaries)]
                       if self._additional_boundaries else [])
            boundary_rows = [[self.__S[row, column] for column in range(self._M)]
                             for row in indices]
            geometry = CollocationMeritGeometry(arithmetic, boundary_rows, self._M)

            def adjoint_direction(gradient):
                try:
                    return geometry.adjoint_direction(gradient)
                except (np.linalg.LinAlgError, ZeroDivisionError) as error:
                    raise LinearSolveError(str(error)) from error

            if self.__use_mpmath:
                # The collocation kernel still uses global mp matrices.
                # Convert clone-owned geometry results back to that matrix
                # class without passing through float64; the public wrapper
                # snapshots all diagnostics in its independent owner later.
                merit_options = dict(
                    merit_gradient=lambda residual, matrix: self.__mp_ref.matrix(tuple(
                        geometry.merit_gradient(residual, matrix))),
                    dot=lambda left, right: scalar(geometry.dot(left, right)),
                    adjoint_direction=lambda gradient: self.__mp_ref.matrix(tuple(
                        adjoint_direction(gradient))),
                )
            else:
                merit_options = dict(merit_gradient=geometry.merit_gradient,
                                     dot=geometry.dot,
                                     adjoint_direction=adjoint_direction)

        return NonlinearSystem(
            residual=self.__residual_operator,
            jacobian=self.__linearization_operator,
            solve_linear=solve_linear,
            residual_norm=lambda residual: self.__norm(residual)/self.__sqrt(self._M),
            initial_guess=lambda: initial.copy(),
            scalar=scalar,
            isfinite=self.__mp_ref.isfinite,
            **merit_options,
        )

    def reference_operator(self, strength="1"):
        """Return A,b for the normalized screened reference R(c)=A*c+b.

        The same differential operator and tau data are used as in the
        physical discretization; only the source is replaced by strength*u.
        No physical source callback is evaluated here.
        """
        scalar = self.__mp_ref.mpf if self.__use_mpmath else np.float64
        strength = scalar(strength)
        if not self.__mp_ref.isfinite(strength) or strength <= 0:
            raise ValueError("reference strength must be finite and positive")
        scale = self.__problem.left_hand_side()[2]
        scale = scalar(1) if scale is None else scalar(scale)
        reaction = self.mass_matrix()*strength*scale
        reaction = self.linear_functional_linearization_boundary(reaction)
        matrix = reaction-self.__S
        if self.__use_mpmath:
            offset = self.__mp_ref.zeros(self._M, 1)
        else:
            offset = np.zeros(self._M, dtype=np.float64)
        offset = self.linear_functional_boundary(offset)
        return matrix, offset

    def residual_L2_norm(self, c):
        relnorm = self.__norm(self.__residual_operator(c))/self.__sqrt(self._M)
        return relnorm

    def solve_problem(self, c0 = None):
        self.__converged = False
        self.__newton_result = None
        self.__initialization_result = None
        self.__scipy_result = None
        try:
            if self.__owns_reporter:
                self.__progress = ProgressReporter(
                    arithmetic=self.__reporting_arithmetic, solver="collocation",
                    method=self.__use_method, **self.__reporter_options).bind(norm="rms")
                self.__progress.emit(
                    "solve_start", N=self._N, unknown_count=self._M,
                    arithmetic="mpmath" if self.__use_mpmath else "numpy",
                    backend=self.__reporting_arithmetic.backend,
                    precision=self.__reporting_arithmetic.dps)
            coefficients = self.__solve_problem(c0)
        except BaseException as error:
            if self.__owns_reporter:
                self.__progress.emit("solve_error", converged=False,
                                     error_type=type(error).__name__, message=str(error))
            raise
        if self.__owns_reporter:
            diagnostics = self.__newton_result or self.__scipy_result
            self.__progress.emit(
                "solve_end", converged=self.__converged,
                nonlinear_converged=self.__converged,
                residual_norm=getattr(diagnostics, "residual_norm", self.__final_residual_norm),
                reason=getattr(diagnostics, "reason", "converged" if self.__converged
                               else "not_converged"),
                homotopy=getattr(diagnostics, "homotopy",
                                 self.__reporting_arithmetic.scalar(0)))
        return coefficients

    def __solve_problem(self, c0=None):
        if self.__problem==None:
            raise ValueError("one should call 'set_problem' before attempring to solve one.")
        validate = getattr(self.__problem, "validate_collocation_compatibility", None)
        if validate is not None:
            validate()
        converged = False
        if c0 is not None:
            self.__base_solution = c0   
        with self.__progress.phase("initial_guess", level=1):
            if self.__base_solution is None:
                c0 = self.__base_solution_expand()
            else:
                c0 = self.function_expand(
                    lambda x: self.solution_in_domain(self.__base_solution, x),
                    interior_only=isinstance(self.__problem, CollocationProblemAdapter),
                )
        c = c0;

        if self.__initializer == "fsolve":
            initial_system = NonlinearSystem(
                residual=self.__residual_operator, jacobian=self.__linearization_operator,
                solve_linear=self.solve,
                residual_norm=lambda residual: self.__norm(residual)/self.__sqrt(self._M),
                initial_guess=lambda: c.copy(), scalar=np.float64, isfinite=np.isfinite,
            )
            self.__initialization_result = initialize_fsolve(
                initial_system, c.copy(), tolerance=self.__initializer_tolerance,
                max_evaluations=self.__initializer_max_evaluations,
                progress=self.__progress,
            )
            c = self.__initialization_result.state.copy()

        # Keep "nonlin_solve" as an alias for the public SciPy root solver.
        if self.__use_method in ("root", "root_no_jac", "nonlin_solve"):
            if self.__use_mpmath:
                raise ValueError("only 'newton' can be used to solve with mpmath library")
            target = np.float64(self.__tolerance)*self.__sqrt(self._M)
            evaluations = [0, 0]
            best_norm = np.float64("inf")

            def scipy_residual(coefficients):
                nonlocal best_norm
                values = self.__residual_operator(coefficients)
                evaluations[0] += 1
                if self.__progress.wants("scipy_evaluation", level=2, throttle=True):
                    if not np.all(np.isfinite(values)):
                        norm = np.float64("inf")
                    else:
                        maximum = np.max(np.abs(values))
                        with np.errstate(under="ignore"):
                            norm = (np.float64(0) if maximum == 0 else
                                    maximum*np.sqrt(np.mean((values/maximum)**2)))
                    best_norm = min(best_norm, norm)
                    self.__progress.emit(
                        "scipy_evaluation", level=2, throttle=True,
                        function_evaluations=evaluations[0],
                        jacobian_evaluations=evaluations[1], residual_norm=norm,
                        best_reported_residual_norm=best_norm, target_norm=target,
                        homotopy=np.float64(0))
                return values

            def scipy_jacobian(coefficients):
                with self.__progress.phase("scipy_jacobian", level=3,
                                           jacobian_evaluations=evaluations[1]+1):
                    matrix = self.__linearization_operator(coefficients)
                evaluations[1] += 1
                return matrix

            options = {} if self.__use_method == "root_no_jac" else {"jac": scipy_jacobian}
            with self.__progress.phase("scipy_solve", level=1, homotopy=np.float64(0),
                                       target_norm=target):
                res = optimize.root(scipy_residual, c, method="hybr", tol=target, **options)
            self.__progress.emit(
                "scipy_stop", scipy_success=bool(getattr(res, "success", False)),
                status=getattr(res, "status", None), message=str(getattr(res, "message", "")),
                nfev=getattr(res, "nfev", None), njev=getattr(res, "njev", None))
            with self.__progress.phase("scipy_verification", level=1,
                                       homotopy=np.float64(0)):
                self.__scipy_result = verify_scipy_result(
                    res, residual=lambda coefficients: self.__residual_operator(coefficients, q=0),
                    target_norm=target, method=self.__use_method,
                )
            c = res.x
            converged = self.__scipy_result.converged
            self.__progress.emit(
                "scipy_verification", converged=self.__scipy_result.converged,
                reason=self.__scipy_result.reason,
                residual_norm=self.__scipy_result.residual_norm,
                residual_max=self.__scipy_result.residual_max, target_norm=target,
                homotopy=self.__scipy_result.homotopy)
        elif self.__use_method == "newton":
            scalar = self.__mp_ref.mpf if self.__use_mpmath else np.float64

            def iteration_callback(coefficients, iteration):
                if self.__visualize:
                    t = np.arange(0, np.pi, 0.01)
                    u = self.obtain_solution_in_basis(coefficients, t)
                    plt.plot(t, u, label=iteration.iteration)

            system = self.nonlinear_system(c)
            # Preserve the historical RMS stopping threshold exposed by tolerance.
            target = scalar(self.__tolerance)*self.__sqrt(self._M)
            self.__newton_result = solve_continuation(
                system,
                reference_factory=self.reference_operator,
                options=self.__continuation,
                atol=target,
                rtol=0,
                max_iterations=self.__total_iterations,
                use_globalization=self.__use_globalization,
                newton_fallback=self.__newton_fallback,
                adjoint_max_steps=self.__adjoint_max_steps,
                callback=iteration_callback,
                progress=self.__progress,
            )
            c = self.__newton_result.state
            converged = self.__newton_result.converged
            if self.__visualize:
                plt.legend()
                plt.show()

        elif self.__use_method == "print":
            k = np.arange(0, self._M)
            j = self.all_discrete_points_in_basis_with_bounds()
            print(j[self._M-2])
            print(self.dddpsi(k, j[self._M-1] ))
            print(sum(self.dddpsi(k, j[self._M-1] ) ))
            print(self.dddpsi(10, j[self._M-1] ))

        if self.__scipy_result is not None:
            final_norm = self.__scipy_result.residual_norm
        else:
            try:
                final_norm = self.residual_L2_norm(c)
            except OverflowError:
                # Newton already classifies overflowing physical callbacks as a
                # numerical failure. Reporting that failure must not raise again.
                if self.__newton_result is None:
                    raise
                final_norm = self.__mp_ref.inf
        self.__final_residual_norm = final_norm
            
        self.__current_solution = c
        self.__converged = converged
        return c

    def is_converged(self):
        return self.__converged

    def get_newton_result(self):
        """Return Newton stage/step diagnostics, or None for other methods."""
        return self.__newton_result

    def get_initialization_result(self):
        """Return optional initial-guess refinement diagnostics."""
        return self.__initialization_result

    def get_scipy_result(self):
        """Return SciPy termination and independent residual diagnostics."""
        return self.__scipy_result

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
            coefficients = [self.__coefficient_as_mpf(value)
                            for value in np.asarray(data, dtype=object).reshape(-1)]
            data_1 = {"format_version": 1, "solver_type": "collocation", "backend": "mpmath",
                      "precision_dps": self.__mp_ref.mp.dps, "coefficients": coefficients}
        else:
            data_1 = {"format_version": 1, "solver_type": "collocation", "backend": "numpy",
                      "coefficients": np.asarray(data, dtype=np.float64).reshape(-1)}
        with open(file_name, 'wb') as handle:
            pickle.dump(data_1, handle)

    def __coefficient_as_mpf(self, value):
        # Reconstructing an existing mpf would round it to the current precision.
        if type(value) is self.__mp_ref.mpf:
            return value
        if isinstance(value, (float, np.floating)):
            # Keep the stored binary value when importing legacy extended precision.
            if not np.isfinite(value):
                return self.__mp_ref.mpf(str(value))
            numerator, denominator = value.as_integer_ratio()
            precision = max(self.__mp_ref.mp.prec, np.finfo(type(value)).nmant + 1)
            with self.__mp_ref.workprec(precision):
                return self.__mp_ref.mpf(numerator) / denominator
        return self.__mp_ref.mpf(value)

    def __load_data(self, file_name):
        with open(file_name, 'rb') as handle:
            prefix = handle.read(256).lstrip()
            handle.seek(0)
            if prefix.startswith(b"{"):
                import json
                data = json.load(handle)
                if not isinstance(data, dict) or data.get("solver_type") != "collocation":
                    raise ValueError("The collocation reader cannot load a different solver type.")
                raise ValueError("Use overscreening_breakdown.load_solution with the physical problem for JSON solutions.")
            data = pickle.load(handle)
        if isinstance(data, dict):
            if data.get("solver_type", "collocation") != "collocation":
                raise ValueError("The collocation reader cannot load a different solver type.")
            if data.get("format_version") != 1:
                raise ValueError("Unsupported solution file format version.")
            if data.get("backend") not in ("numpy", "mpmath") or "coefficients" not in data:
                raise ValueError("Invalid solution file format.")
            data = data["coefficients"]
        # Older files contain a bare NumPy array, often with dtype float128.
        if self.__use_mpmath:
            coefficients = [self.__coefficient_as_mpf(value)
                            for value in np.asarray(data, dtype=object).reshape(-1)]
            data_1 = self.__mp_ref.matrix(coefficients)
        else:
            data_1 = np.asarray(data, dtype=np.float64).reshape(-1)
        return data_1

    def save_solution(self, file_name, c):
        """Save coefficient values and backend metadata without reducing mpmath precision."""
        self.__save_data(file_name, c)

    def load_data(self, file_name):
        """Load a coefficient vector in the active backend, including legacy array files."""
        c = self.__load_data(file_name)
        return c

    def set_base_solution(self, file_name):
        self.__base_solution = self.__load_data(file_name)

    def reset_base_solution(self):
        self.__base_solution = None
