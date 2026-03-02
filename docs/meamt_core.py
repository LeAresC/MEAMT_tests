import random
import numpy as np
from deap import creator, base, tools, benchmarks
from deap.benchmarks.tools import hypervolume

# ==========================================
# 1. SETUP DE CLASSES E TOOLBOX
# ==========================================
def setup_deap_classes(n_obj):
    """Inicializa as classes do DEAP com base no número de objetivos."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin, Parent_Table=None)
        creator.create("SubPopulation", list, score=0.0)

def build_toolbox(funcao_avaliacao, ind_size, n_pop, n_obj):
    """Constrói o toolbox do DEAP de forma dinâmica para qualquer benchmark."""
    setup_deap_classes(n_obj)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=n_pop)
    
    # Avaliação dinâmica (passada por parâmetro)
    toolbox.register("evaluate", funcao_avaliacao) 
    
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=0.0, up=1.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0/ind_size)
    
    return toolbox

# ==========================================
# 2. MÉTRICAS E FRONTEIRAS
# ==========================================
import numpy as np

def generate_zdt3_front_true(n_points=10000):
    """
    Gera a fronteira de Pareto verdadeira para o ZDT3.
    """
    # Para o ZDT3, f1 varia de 0 a 1 e g(x) = 1 na fronteira ótima
    f1 = np.linspace(0, 1, n_points)
    
    # Equação do f2 para o ZDT3
    f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
    points = np.column_stack((f1, f2))
    
    # O ZDT3 é descontínuo. Precisamos filtrar os pontos dominados
    non_dominated = []
    min_f2 = float('inf')
    
    # Como f1 já está ordenado de forma crescente, 
    # basta garantir que f2 está sempre diminuindo
    for p in points:
        if p[1] < min_f2:
            non_dominated.append(p)
            min_f2 = p[1]
            
    return np.array(non_dominated)
def generate_dtlz3_front_random(n_obj, n_points):
    """Generates points on the DTLZ3 Pareto front (Unit Hypersphere)."""
    samples = np.abs(np.random.normal(size=(n_points, n_obj)))
    radius = np.sqrt(np.sum(samples**2, axis=1, keepdims=True))
    pf = samples / radius
    return pf

def calculate_igd_plus(pareto_front_true, pareto_front_approx):
    """Calculates the IGD+ (Inverted Generational Distance Plus)."""
    pf_true = np.atleast_2d(pareto_front_true)
    pf_approx = np.atleast_2d(pareto_front_approx)
    dists = []

    for z in pf_true:
        diff = pf_approx - z
        diff = np.maximum(diff, 0)
        d_plus = np.sqrt(np.sum(diff**2, axis=1))
        min_d_plus = np.min(d_plus)
        dists.append(min_d_plus)

    return np.mean(dists)

# ==========================================
# 3. OPERADORES DO MEAMT
# ==========================================
def calc_combined_fitness(ind, table_idx, n_obj):
    """Calculates the combined fitness of an individual with respect to his table."""
    fit = 0
    for b in range(n_obj):
        # Desloca os bits e verifica se o bit na posição 'b' é 1
        if (table_idx >> b) & 1:
            # Pega o valor correspondente de trás pra frente (como no seu original)
            fit += ind.fitness.values[n_obj - 1 - b]
    return fit

def gen_inicial_tables(pop_ini, num_tables, table_size, n_obj):
    """Inserts inicial population in the tables."""
    tables = dict()
    # Tabel 0: ND (Non Dominated)
    fronteira = tools.sortNondominated(pop_ini, len(pop_ini), first_front_only=True)[0]
    tables[0] = creator.SubPopulation(fronteira[:table_size]) 
    tables[0].score = 0.0

    # Tabel 1 to N
    for i in range(1, num_tables):
        # Repassando n_obj para o lambda
        pop_ordenada = sorted(pop_ini, key=lambda ind: calc_combined_fitness(ind, i, n_obj))
        tables[i] = creator.SubPopulation(pop_ordenada[:table_size])
        tables[i].score = 0.0
    return tables

def select_parents(tables, num_tables):
    """Selects parents based on table scores."""
    selected = []
    for _ in range(2):
        random1 = random.randint(0, num_tables - 1)
        random2 = random.randint(0, num_tables - 1)

        if len(tables[random1]) == 0: winner = random2
        elif len(tables[random2]) == 0: winner = random1
        elif tables[random1].score >= tables[random2].score:
            winner = random1
        else:
            winner = random2

        ind = random.choice(tables[winner])
        ind.Parent_Table = winner
        selected.append(ind)
    return selected

def insert_in_tables(tables, num_tables, off, max_table_size, n_obj):
    """Inserts offspring into the appropriate tables and updates scores."""
    # 1. Insert on table ND (0)
    tabela_nd = tables[0]
    tabela_nd.append(off)
    
    # NSGA2 select
    nova_selecao = tools.selNSGA2(tabela_nd, max_table_size)
    tabela_nd[:] = nova_selecao

    # Se off is in the table -> increase score
    if any(ind is off for ind in tabela_nd):
        if off.Parent_Table is not None:
            tables[off.Parent_Table].score += 1

    # Try to insert in other tables
    for i in range(1, num_tables):
        # Repassando o n_obj aqui
        fit_off = calc_combined_fitness(off, i, n_obj)
        worse_val = -1.0
        worse_idx = -1

        for idx, ind in enumerate(tables[i]):
            # Repassando o n_obj aqui
            fit_ind = calc_combined_fitness(ind, i, n_obj)
            if fit_ind > worse_val:
                worse_val = fit_ind
                worse_idx = idx

        # Minimization: If new is less than the worst, replaces it
        if fit_off < worse_val:
            tables[i][worse_idx] = off
            if off.Parent_Table is not None:
                tables[off.Parent_Table].score += 1

# ==========================================
# 4. LOOP PRINCIPAL
# ==========================================
def run(tables, pareto_front_true, num_tables, max_table_size, ngen, toolbox, cxpb, mutpb, ref_point_hv, n_obj):
    """Executa as gerações do algoritmo MEAMT (Versão Paralela)."""
    logbook = tools.Logbook()
    logbook.header = "gen", "hypervolume", "igd_plus"
    
    for gen in range(1, ngen + 1):
        # Reset score
        for t in tables.values():     
            t.score = 0.0

        # =========================================================
        # FASE 1: GERAR TODOS OS FILHOS DA GERAÇÃO
        # =========================================================
        offspring = []
        for _ in range((max_table_size * num_tables) // 2):
            parents = select_parents(tables, num_tables)

            off1, off2 = toolbox.clone(parents[0]), toolbox.clone(parents[1])
            off1.Parent_Table = parents[0].Parent_Table
            off2.Parent_Table = parents[1].Parent_Table

            if random.random() < cxpb:
                toolbox.mate(off1, off2)
                del off1.fitness.values, off2.fitness.values

            if random.random() < mutpb:
                toolbox.mutate(off1)
                del off1.fitness.values
            if random.random() < mutpb:
                toolbox.mutate(off2)
                del off2.fitness.values
                
            offspring.extend([off1, off2])

        # =========================================================
        # FASE 2: AVALIAÇÃO EM PARALELO (A Mágica acontece aqui!)
        # =========================================================
        # Separa apenas os indivíduos que sofreram mutação/crossover e precisam de avaliação
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        # O toolbox.map vai distribuir a lista de indivíduos entre os núcleos do seu PC
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        # Atribui os resultados de volta aos indivíduos
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # =========================================================
        # FASE 3: INSERÇÃO
        # =========================================================
        for off in offspring:
            insert_in_tables(tables, num_tables, off, max_table_size,n_obj)
            
        # =========================================================
        # MÉTRICAS
        # =========================================================
        hv_val = hypervolume(tables[0], ref_point_hv)
        approx_front = np.array([ind.fitness.values for ind in tables[0]])
        igd_plus_val = calculate_igd_plus(pareto_front_true, approx_front)
        
        logbook.record(gen=gen, hypervolume=hv_val, igd_plus=igd_plus_val)
        
    return logbook