from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sympy import symbols, integrate, simplify, Eq, solve, sympify, diff
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
import time
import traceback

app = FastAPI(title="Gibbs Sampler API", version="2.0.0")

# Configurar CORS para permitir requests del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar el directorio 'static' para servir archivos estáticos
app.mount("/static", StaticFiles(directory="Simulador/static"), name="static")

# Serve la página principal
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("Simulador/prueba.html", "r", encoding="utf-8") as f:
        return f.read()

# Símbolos globales - usar estos en toda la aplicación
x, y, u = symbols('x y u', real=True)

class GibbsRequest(BaseModel):
    expression: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_initial: Optional[float] = None
    y_initial: Optional[float] = None
    n_samples: int = 1000
    burn_in: int = 500

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_expression: Optional[str] = None

class GibbsResult(BaseModel):
    success: bool
    validation: ValidationResult
    samples: Optional[Dict[str, List[float]]] = None
    statistics: Optional[Dict] = None
    execution_time: Optional[float] = None
    plot_data: Optional[Dict] = None

class GibbsSampler:
    def __init__(self):
        # NO crear nuevos símbolos, usar los globales
        pass
    
    def validate_expression(self, expr_str: str, x_bounds: Tuple[float, float], 
                          y_bounds: Tuple[float, float]) -> ValidationResult:
        """Valida si la expresión puede ser procesada analíticamente"""
        errors = []
        warnings = []
        normalized_expr = None
        
        try:
            # 1. Parsear la expresión usando los símbolos globales
            # Crear un diccionario local de transformaciones para el parsing
            local_dict = {'x': x, 'y': y}
            
            try:
                expr = parse_expr(expr_str, local_dict=local_dict, transformations='all')
            except:
                # Si falla el parsing avanzado, intentar parsing básico
                expr = sympify(expr_str, locals=local_dict)
            
            normalized_expr = str(expr)
            print(f"Expresión parseada: {expr}")
            print(f"Símbolos libres: {expr.free_symbols}")
            
            # 2. Verificar que solo contenga x, y y constantes
            free_symbols = expr.free_symbols
            allowed_symbols = {x, y}  # Usar símbolos globales
            invalid_symbols = free_symbols - allowed_symbols
            
            if invalid_symbols:
                errors.append(f"Símbolos no permitidos: {[str(s) for s in invalid_symbols]}")
            
            # 3. Verificar que la función sea no negativa en el dominio
            test_points = [
                (x_bounds[0], y_bounds[0]),
                (x_bounds[1], y_bounds[1]),
                ((x_bounds[0] + x_bounds[1])/2, (y_bounds[0] + y_bounds[1])/2),
                (x_bounds[0], y_bounds[1]),
                (x_bounds[1], y_bounds[0])
            ]
            
            for px, py in test_points:
                try:
                    val = float(expr.subs({x: px, y: py}))
                    if val < 0:
                        errors.append(f"Función negativa en ({px}, {py}): {val}")
                        break
                except Exception as e:
                    errors.append(f"Error evaluando función en ({px}, {py}): {str(e)}")
                    break
            
            # 4. Calcular marginales
            try:
                print("Calculando marginales...")
                f_x = integrate(expr, (y, y_bounds[0], y_bounds[1]))
                f_y = integrate(expr, (x, x_bounds[0], x_bounds[1]))
                
                print(f"Marginal f_X(x): {f_x}")
                print(f"Marginal f_Y(y): {f_y}")
                
                if f_x == 0 or f_y == 0:
                    errors.append("Una de las distribuciones marginales es cero")
                
                # Verificar que las marginales sean funciones de la variable correcta
                if len(f_x.free_symbols) > 1 or (len(f_x.free_symbols) == 1 and x not in f_x.free_symbols):
                    errors.append("Error en marginal f_X(x): contiene símbolos incorrectos")
                
                if len(f_y.free_symbols) > 1 or (len(f_y.free_symbols) == 1 and y not in f_y.free_symbols):
                    errors.append("Error en marginal f_Y(y): contiene símbolos incorrectos")
                
            except Exception as e:
                errors.append(f"Error calculando marginales: {str(e)}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
            
            # 5. Verificar si se pueden calcular condicionales
            try:
                print("Calculando distribuciones condicionales...")
                fx_given_y = simplify(expr / f_y)
                fy_given_x = simplify(expr / f_x)
                
                print(f"f(x|y): {fx_given_y}")
                print(f"f(y|x): {fy_given_x}")
                
                # Intentar calcular CDFs
                print("Calculando CDFs...")
                cdf_x = integrate(fx_given_y, (x, x_bounds[0], x))
                cdf_y = integrate(fy_given_x, (y, y_bounds[0], y))
                
                print(f"F(x|y): {cdf_x}")
                print(f"F(y|x): {cdf_y}")
                
                # Verificar si se pueden invertir
                print("Verificando si se pueden invertir las CDFs...")
                eq_x = Eq(u, cdf_x)
                eq_y = Eq(u, cdf_y)
                
                sol_x = solve(eq_x, x)
                sol_y = solve(eq_y, y)
                
                print(f"Soluciones para x: {sol_x}")
                print(f"Soluciones para y: {sol_y}")
                
                if not sol_x:
                    errors.append("No se puede invertir F(x|y) - no hay solución analítica")
                if not sol_y:
                    errors.append("No se puede invertir F(y|x) - no hay solución analítica")
                    
                # Verificar que al menos una solución sea válida para cada caso
                valid_x = False
                valid_y = False
                
                for sol in sol_x:
                    if sol.is_real is not False:  # Puede ser True o None (indeterminado)
                        valid_x = True
                        break
                
                for sol in sol_y:
                    if sol.is_real is not False:
                        valid_y = True
                        break
                
                if not valid_x:
                    errors.append("No hay soluciones reales válidas para x")
                if not valid_y:
                    errors.append("No hay soluciones reales válidas para y")
                    
            except Exception as e:
                errors.append(f"Error en distribuciones condicionales: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
            
        except Exception as e:
            errors.append(f"Error parseando expresión: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_expression=normalized_expr
        )
    
    def sample(self, expr_str: str, x_bounds: Tuple[float, float], 
               y_bounds: Tuple[float, float], x_init: float, y_init: float,
               n_samples: int, burn_in: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Ejecuta el muestreador de Gibbs"""
        
        start_time = time.time()
        
        # Parsear expresión usando símbolos globales
        local_dict = {'x': x, 'y': y}
        try:
            expr = parse_expr(expr_str, local_dict=local_dict, transformations='all')
        except:
            expr = sympify(expr_str, locals=local_dict)
        
        print(f"Iniciando muestreo con expresión: {expr}")
        
        # Calcular marginales - usando símbolos globales
        f_y = integrate(expr, (x, x_bounds[0], x_bounds[1]))  # marginal de Y
        f_x = integrate(expr, (y, y_bounds[0], y_bounds[1]))  # marginal de X
        
        print(f"Marginal f_Y(y): {f_y}")
        print(f"Marginal f_X(x): {f_x}")
        
        # Distribuciones condicionales
        fx_given_y = simplify(expr / f_y)  # f(x|y)
        fy_given_x = simplify(expr / f_x)  # f(y|x)
        
        print(f"f(x|y): {fx_given_y}")
        print(f"f(y|x): {fy_given_x}")
        
        # CDFs condicionales
        cdf_x = integrate(fx_given_y, (x, x_bounds[0], x))
        cdf_y = integrate(fy_given_x, (y, y_bounds[0], y))
        
        print(f"F(x|y): {cdf_x}")
        print(f"F(y|x): {cdf_y}")
        
        # Funciones inversas
        eq_x = Eq(u, cdf_x)
        eq_y = Eq(u, cdf_y)
        
        sol_x = solve(eq_x, x)
        sol_y = solve(eq_y, y)
        
        print(f"Soluciones x: {sol_x}")
        print(f"Soluciones y: {sol_y}")
        
        # Tomar la solución válida
        inv_x = None
        inv_y = None
        
        for s in sol_x:
            try:
                # Probar si la solución es válida evaluándola
                test_val = float(s.subs({y: (y_bounds[0] + y_bounds[1])/2, u: 0.5}).evalf())
                if x_bounds[0] <= test_val <= x_bounds[1]:
                    inv_x = s
                    break
            except:
                continue
        
        for s in sol_y:
            try:
                # Probar si la solución es válida evaluándola
                test_val = float(s.subs({x: (x_bounds[0] + x_bounds[1])/2, u: 0.5}).evalf())
                if y_bounds[0] <= test_val <= y_bounds[1]:
                    inv_y = s
                    break
            except:
                continue
        
        if inv_x is None or inv_y is None:
            raise ValueError(f"No se encontraron funciones inversas válidas. inv_x: {inv_x}, inv_y: {inv_y}")
        
        print(f"Usando inv_x: {inv_x}")
        print(f"Usando inv_y: {inv_y}")
        
        # Muestreo
        total_samples = n_samples + burn_in
        samples_x = np.zeros(total_samples)
        samples_y = np.zeros(total_samples)
        
        # Inicializar
        current_x = x_init
        current_y = y_init
        samples_x[0] = current_x
        samples_y[0] = current_y
        
        print(f"Iniciando muestreo con {total_samples} iteraciones...")
        
        for i in range(1, total_samples):
            try:
                # Muestrear X dado Y
                u1 = np.random.uniform(0, 1)
                new_x_expr = inv_x.subs({y: current_y, u: u1})
                new_x = float(new_x_expr.evalf())
                
                # Muestrear Y dado X
                u2 = np.random.uniform(0, 1)  
                new_y_expr = inv_y.subs({x: new_x, u: u2})
                new_y = float(new_y_expr.evalf())
                
                # Validar bounds
                new_x = np.clip(new_x, x_bounds[0], x_bounds[1])
                new_y = np.clip(new_y, y_bounds[0], y_bounds[1])
                
                samples_x[i] = new_x
                samples_y[i] = new_y
                current_x, current_y = new_x, new_y
                
                # Progress check
                if i % 1000 == 0:
                    print(f"Completado {i}/{total_samples} iteraciones")
                
            except Exception as e:
                print(f"Error en iteración {i}: {str(e)}")
                # Usar valores anteriores en caso de error
                samples_x[i] = current_x
                samples_y[i] = current_y
        
        # Descartar burn-in
        final_x = samples_x[burn_in:]
        final_y = samples_y[burn_in:]
        
        end_time = time.time()
        
        # Estadísticas
        stats = {
            "mean_x": float(np.mean(final_x)),
            "mean_y": float(np.mean(final_y)),
            "std_x": float(np.std(final_x)),
            "std_y": float(np.std(final_y)),
            "correlation": float(np.corrcoef(final_x, final_y)[0, 1]),
            "execution_time": end_time - start_time,
            "total_samples": len(final_x)
        }
        
        print(f"Muestreo completado en {stats['execution_time']:.3f}s")
        
        return final_x, final_y, stats

def create_plot_data(x_samples: np.ndarray, y_samples: np.ndarray, 
                    x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> Dict:
    """Prepara los datos para las visualizaciones"""
    
    # Datos para scatter 2D
    scatter_data = {
        "x": x_samples.tolist(),
        "y": y_samples.tolist(),
        "type": "scatter",
        "mode": "markers",
        "marker": {
            "size": 3,
            "opacity": 0.6,
            "color": "blue"
        },
        "name": "Muestras"
    }
    
    # Crear histograma 3D
    n_bins = 20
    
    # Crear bins
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)
    y_edges = np.linspace(y_bounds[0], y_bounds[1], n_bins + 1)
    
    # Calcular histograma 2D
    hist, _, _ = np.histogram2d(x_samples, y_samples, bins=[x_edges, y_edges])
    
    # Preparar datos para superficie 3D
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    X, Y = np.meshgrid(x_centers, y_centers)
    
    histogram_3d = {
        "x": X.tolist(),
        "y": Y.tolist(), 
        "z": hist.T.tolist(),
        "type": "surface",
        "colorscale": "Viridis",
        "name": "Densidad"
    }
    
    return {
        "scatter_2d": scatter_data,
        "histogram_3d": histogram_3d,
        "layout_2d": {
            "title": "Muestras de Gibbs - Vista 2D",
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"}
        },
        "layout_3d": {
            "title": "Histograma de Frecuencia - Vista 3D",
            "scene": {
                "xaxis": {"title": "X"},
                "yaxis": {"title": "Y"},
                "zaxis": {"title": "Frecuencia"}
            }
        }
    }

@app.get("/")
def root():
    return {"message": "Gibbs Sampler API v2.0 - Método Analítico"}

@app.post("/validate", response_model=ValidationResult)
def validate_function(request: GibbsRequest):
    """Valida si una función puede ser procesada analíticamente"""
    sampler = GibbsSampler()
    return sampler.validate_expression(
        request.expression,
        (request.x_min, request.x_max),
        (request.y_min, request.y_max)
    )

@app.post("/sample", response_model=GibbsResult)
def gibbs_sample(request: GibbsRequest):
    """Ejecuta el muestreador de Gibbs y genera datos para visualización"""
    try:
        sampler = GibbsSampler()
        
        # Validar primero
        validation = sampler.validate_expression(
            request.expression,
            (request.x_min, request.x_max),
            (request.y_min, request.y_max)
        )
        
        if not validation.is_valid:
            return GibbsResult(
                success=False,
                validation=validation
            )
        
        # Valores iniciales por defecto
        x_init = request.x_initial if request.x_initial is not None else (request.x_min + request.x_max) / 2
        y_init = request.y_initial if request.y_initial is not None else (request.y_min + request.y_max) / 2
        
        # Ejecutar muestreo
        x_samples, y_samples, stats = sampler.sample(
            request.expression,
            (request.x_min, request.x_max),
            (request.y_min, request.y_max),
            x_init, y_init,
            request.n_samples,
            request.burn_in
        )
        
        # Preparar datos para visualización
        plot_data = create_plot_data(
            x_samples, y_samples,
            (request.x_min, request.x_max),
            (request.y_min, request.y_max)
        )
        
        return GibbsResult(
            success=True,
            validation=validation,
            samples={
                "x": x_samples.tolist(),
                "y": y_samples.tolist()
            },
            statistics=stats,
            execution_time=stats["execution_time"],
            plot_data=plot_data
        )
        
    except Exception as e:
        error_msg = str(e)
        traceback_info = traceback.format_exc()
        
        return GibbsResult(
            success=False,
            validation=ValidationResult(
                is_valid=False,
                errors=[f"Error durante el muestreo: {error_msg}"],
                warnings=[f"Traceback: {traceback_info}"]
            )
        )

@app.get("/examples")
def get_examples():
    """Retorna ejemplos de funciones que funcionan bien"""
    return {
        "examples": [
            {
                "name": "Distribución Lineal",
                "expression": "(2*x + 3*y + 2)/28",
                "bounds": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2},
                "description": "Función de densidad conjunta lineal"
            },
            {
                "name": "Distribución Cuadrática",
                "expression": "(x**2 + y**2 + 1)/21",
                "bounds": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2},
                "description": "Función cuadrática simple"
            },
            {
                "name": "Distribución Mixta",
                "expression": "(x + 2*y**2 + 3)/15",
                "bounds": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
                "description": "Combinación lineal-cuadrática"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)