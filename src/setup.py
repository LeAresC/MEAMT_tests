import random
from deap import creator, base, tools, benchmarks
from deap.benchmarks.tools import hypervolume
import matplotlib.pyplot as plt
# --- CONFIGURAÇÕES ---
IND_SIZE = 12
NPOP = 640
NGEN = 500
NOBJ = 3
CXPB = 0.9
MUTPB = 1.0
NUM_TABLES = int((1 << NOBJ))
MAX_TABLE_SIZE = int(NPOP / NUM_TABLES)

# --- CREATORS ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin, Parent_Table=None)
creator.create("SubPopulation", list, score=0.0) 

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=NPOP)
toolbox.register("evaluate", benchmarks.dtlz3, obj=NOBJ)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=0.0, up=1.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0/IND_SIZE)

REF_POINT_HV = [1.1] * 3

# --- FUNÇÃO AUXILIAR PARA CALCULAR FITNESS DA TABELA ---
def calc_combined_fitness(ind, table_idx):
    # Essa função substitui aquela linha longa e previne erros de índice
    fit = 0
    # Bit 0 -> Obj 2 | Bit 1 -> Obj 1 | Bit 2 -> Obj 0 (ou qualquer ordem consistente)
    # Apenas garantindo que usamos índices diferentes para bits diferentes
    if (table_idx >> 0) & 1: fit += ind.fitness.values[2]
    if (table_idx >> 1) & 1: fit += ind.fitness.values[1]
    if (table_idx >> 2) & 1: fit += ind.fitness.values[0]
    return fit

def gen_inicial_tables(pop_ini, num_tables, table_size):
    tables = dict()
    # Tabela 0: ND (Não Dominados)
    fronteira = tools.sortNondominated(pop_ini, len(pop_ini), first_front_only=True)[0]
    tables[0] = creator.SubPopulation(fronteira[:table_size]) # Corta se for maior
    tables[0].score = 0.0

    # Tabelas 1 a 7
    for i in range(1, num_tables):
        # Usa a função auxiliar corrigida
        pop_ordenada = sorted(pop_ini, key=lambda ind: calc_combined_fitness(ind, i))
        tables[i] = creator.SubPopulation(pop_ordenada[:table_size])
        tables[i].score = 0.0
    return tables

def select_parents(tables, num_tables):
    selected = []
    for _ in range(2):
         random1 = random.randint(0, num_tables - 1)
         random2 = random.randint(0, num_tables - 1)
         
         # Proteção para tabela vazia (caso raro)
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

def insert_in_tables(tables, num_tables, off, max_table_size):
    # 1. Tenta inserir na Tabela ND (0)
    tabela_nd = tables[0]
    tabela_nd.append(off)
    # NSGA2 seleciona os melhores e corta o excesso
    nova_selecao = tools.selNSGA2(tabela_nd, max_table_size)
    tabela_nd[:] = nova_selecao
    
    # Se off está na nova seleção, ele sobreviveu -> Ponto para o pai!
    if any(ind is off for ind in tabela_nd):
        if off.Parent_Table is not None:
            tables[off.Parent_Table].score += 1

    # 2. Tenta inserir nas Tabelas de Critério (1 a 7)
    for i in range(1, num_tables):
        fit_off = calc_combined_fitness(off, i)
        
        # Encontra o PIOR (maior valor) da tabela atual
        worse_val = -1.0
        worse_idx = -1
        
        for idx, ind in enumerate(tables[i]):
            fit_ind = calc_combined_fitness(ind, i)
            if fit_ind > worse_val:
                worse_val = fit_ind
                worse_idx = idx
        
        # Minimização: Se novo é MENOR que o PIOR, substitui
        if fit_off < worse_val:
            tables[i][worse_idx] = off
            if off.Parent_Table is not None:
                tables[off.Parent_Table].score += 1

def run(tables, num_tables, max_table_size, ngen, toolbox, cxpb, mutpb):  
    logbook = tools.Logbook()
    logbook.header = "gen", "hypervolume"
    for gen in range(1, ngen + 1):
        for t in tables.values():     t.score = 0.0
        
        for _ in range((max_table_size * num_tables) // 2): 
             parents = select_parents(tables, num_tables)
             
             off1 = toolbox.clone(parents[0])
             off2 = toolbox.clone(parents[1])
             
             # Precisamos garantir que o Parent_Table sobreviva ao clone corretamente
             off1.Parent_Table = parents[0].Parent_Table
             off2.Parent_Table = parents[1].Parent_Table

             # Crossover
             if random.random() < cxpb:
                toolbox.mate(off1, off2)
                del off1.fitness.values
                del off2.fitness.values
             
             # Mutação
             if random.random() < mutpb:
                toolbox.mutate(off1)
                del off1.fitness.values
             if random.random() < mutpb:   
                toolbox.mutate(off2)
                del off2.fitness.values
             
             # Avaliação (apenas se necessário)
             if not off1.fitness.valid:
                 off1.fitness.values = toolbox.evaluate(off1)
             if not off2.fitness.valid:
                 off2.fitness.values = toolbox.evaluate(off2)
             
             # Inserção
             insert_in_tables(tables, num_tables, off1, max_table_size)
             insert_in_tables(tables, num_tables, off2, max_table_size)
        hv_val = hypervolume(tables[0], REF_POINT_HV)
        logbook.record(gen=gen,hypervolume=hv_val)
        print(logbook.stream)
    return logbook

# --- MAIN ---
pop_ini = toolbox.population()
for ind in pop_ini:
    ind.fitness.values = toolbox.evaluate(ind)

tables = gen_inicial_tables(pop_ini, NUM_TABLES, MAX_TABLE_SIZE)
logbook = run(tables, NUM_TABLES, MAX_TABLE_SIZE, NGEN, toolbox, CXPB, MUTPB)

# Extrair dados
f1 = [ind.fitness.values[0] for ind in tables[0]]
f2 = [ind.fitness.values[1] for ind in tables[0]]
f3 = [ind.fitness.values[2] for ind in tables[0]]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotar pontos
ax.scatter(f1, f2, f3, c='r', marker='o', label='Soluções Encontradas')

# Configurar eixos (DTLZ3 normalizado vai de 0 a 1)
ax.set_xlabel('Obj 1')
ax.set_ylabel('Obj 2')
ax.set_zlabel('Obj 3')
ax.set_title('Fronteira de Pareto Encontrada (MEAMT - DTLZ3)')
ax.view_init(elev=30, azim=45) # Muda o ângulo da câmera
plt.show()

gen = logbook.select("gen")
fit_hv = logbook.select("hypervolume")
fig, ax1 = plt.subplots(1, 1, figsize=(18, 5))
ax1.plot(gen, fit_hv, 'b-')
ax1.set_title("Hypervolume (Maximizar)")
ax1.set_xlabel("Geração")
ax1.grid()
plt.tight_layout()
plt.show()