# Teoria do Débito Temporal (TDT)

Pesquisa computacional independente sobre irreversibilidade, memória, informação e controle em sistemas físicos finitos.

**Natureza do trabalho:** estrutura teórica exploratória, acompanhada por simulações clássicas e quânticas. A TDT não é apresentada aqui como lei física estabelecida nem como teoria universal validada.

## A pergunta central

A pesquisa investiga se descrições operacionais de irreversibilidade e memória podem ajudar a caracterizar o custo de transformar, controlar ou recuperar estados físicos quando parte da informação relevante permanece distribuída pelo ambiente.

A ideia foi examinada em modelos diferentes — sistemas de dois níveis, cadeias de Ising, banhos finitos, modelos quânticos com modos explícitos e processos markovianos e não markovianos. Cada modelo testa apenas uma parte da hipótese; nenhum resultado isolado demonstra a teoria inteira.

## Linhas de investigação

- **Termodinâmica da informação:** limites de Landauer, apagamento e informação lateral.
- **Banhos finitos e recuperação:** modelos de Ising e demônio de Creutz, com simulações estocásticas.
- **Memória quântica:** modos harmônicos explícitos, dinâmica reduzida e testemunhas de backflow.
- **Controle ótimo:** protocolos numéricos, refinamento temporal e comparação com referências convencionais.

## Alguns resultados que merecem atenção

- Um benchmark de dois níveis resolve a dinâmica mestra numericamente e verifica o balanço de energia com resíduos próximos da precisão numérica. Ao mesmo tempo, a quantidade operacional denominada Pi varia com a escala de referência e não coincide, em geral, com a dissipação recuperável. Isso delimita uma questão matemática importante.
- Um modelo com dois qubits e dois modos harmônicos truncados tem dimensão de Hilbert 36 e apresenta backflow de distância de traço para o par de estados e os parâmetros escolhidos. Esse resultado caracteriza memória no modelo específico; não é uma assinatura exclusiva da TDT.
- Uma linha de simulações estocásticas de banhos finitos executa até 100.000 trajetórias por condição para comparar estratégias de controle, evolução e recuperação em modelos de Ising e do demônio de Creutz.
- Um cálculo exato de um anel de Ising de quatro bits estuda o limite ideal de apagamento quando a magnetização final é disponibilizada como informação lateral.

Os detalhes, arquivos de resultado e limitações estão em [docs/EXPERIMENT_CATALOG.md](docs/EXPERIMENT_CATALOG.md).

## Organização do laboratório

- `*.py` — simulações e análises em Python.
- `*.cpp` — simuladores compilados para experimentos estocásticos.
- `*.json` — resultados estruturados produzidos por execuções específicas.
- `*.txt` e `*.log` — relatórios e registros de execução.
- `*.npz`, `*.csv` e `*.png` — artefatos numéricos e visuais de experimentos selecionados.

Os nomes `v2`, `v4` e `v5` identificam etapas históricas de investigação, não versões de uma teoria já consolidada.

## Ambiente

- Python 3.12 ou superior.
- NumPy e SciPy para as simulações numéricas.
- Matplotlib para scripts de visualização.
- Compilador C++ compatível para os executáveis em `*.cpp`.

Crie um ambiente virtual antes de instalar as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Reprodução:** consulte primeiro o catálogo de experimentos. Alguns scripts históricos usam caminhos locais absolutos ou gravam resultados no diretório de trabalho. Execute-os em uma cópia limpa do laboratório para preservar os resultados já registrados.

## Integridade científica

- Resultados negativos, inconclusivos e limitações fazem parte do histórico.
- Backflow não implica, por si só, uma nova lei termodinâmica.
- Otimização por SPSA não prova que um mínimo global foi encontrado.
- Modelos Markovianos e modelos com modos explícitos só podem ser comparados quando suas hipóteses e parâmetros físicos forem compatíveis.
- Uma identidade numérica ou um balanço de energia correto não demonstra que a TDT acrescente uma previsão nova.
- A formulação matemática central ainda precisa de definições operacionais consistentes, análise dimensional e testes independentes.

## Estado atual

A TDT permanece como programa de pesquisa exploratório. O valor do repositório está no conjunto de modelos, códigos, resultados rastreáveis e perguntas que sobreviveram — ou não — aos testes realizados. Não há alegação de validação experimental, consenso científico ou aplicabilidade universal.

---

Leia também: [English](README.en.md) · [Español](README.es.md) · [Catálogo de experimentos](docs/EXPERIMENT_CATALOG.md)