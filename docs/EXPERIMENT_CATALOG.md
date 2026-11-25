# Catálogo de experimentos selecionados

Este catálogo destaca estudos que ajudam a entender o alcance da TDT e os limites encontrados. Ele não é uma lista de vitórias: registra o que cada simulação permite concluir, e o que não permite.

## 1. Sistema de dois níveis — controle e trabalho

**Arquivos:** `tdt_v2_two_level_optimal_control.py`, `tdt_v2_two_level_optimal_control.txt` e `tdt_v2_two_level_optimal_control.json`.

Benchmark de equação mestra com dinâmica numérica para um sistema de dois níveis, protocolo discretizado e otimização do controle. O balanço de primeira lei fecha com resíduos da ordem de 10⁻¹⁵ no registro salvo.

O resultado também mostra um limite importante: a quantidade operacional denominada Pi varia com a escala de referência escolhida (0,1; 1; 10 kBT). Para o caminho direto, os valores registrados são aproximadamente 3,872; 1,741; e 0,385 kBT, enquanto a dissipação de recuperação otimizada é aproximadamente 0,703 kBT. Portanto, a identificação entre Pi e dissipação não é demonstrada por esse benchmark.

## 2. Landauer com informação lateral

**Arquivo:** `landauer_4bit_conditional_exact.py` e o JSON correspondente.

Modelo de anel de Ising com quatro bits, 16 estados e dinâmica markoviana. O script calcula a distribuição conjunta entre o microestado inicial e a magnetização final, compara entropias condicionais e verifica identidades do limite ideal de Landauer.

É um cálculo exato e útil para estudar o papel da informação lateral no apagamento. Não é, por si só, uma confirmação da TDT: o limite de Landauer e as identidades de informação usados pertencem à termodinâmica da informação convencional.

## 3. Banho finito de Creutz — recuperação exata

**Arquivos:** `tdt_v4_creutz40_exact_recovery.cpp`, `tdt_v4_creutz40_exact_recovery.txt` e variantes de H.

Simulação estocástica de um anel de 40 spins, com subsistema de 8 spins e banho de 32 spins. O registro principal usa 100.000 trajetórias por condição e compara controle restrito ao subsistema com controle aplicado ao anel inteiro.

Em várias condições com evolução e controle, a recuperação exata observada foi 0/100.000. No controle de todos os 40 spins sem etapa prévia de evolução, houve 121/100.000 recuperações; nas etapas de evolução seguintes, a contagem voltou a zero. Isso não demonstra impossibilidade geral de recuperação: mostra que os protocolos e parâmetros específicos testados não alcançaram o objetivo.

## 4. Dois qubits e dois modos explícitos — memória não markoviana

**Arquivos:** `tdt_quantum_nonmarkovian_2mode.py`, `tdt_quantum_nonmarkovian_2mode_result.json` e `tdt_quantum_nonmarkovian_2mode_strong_result.json`.

O modelo tem dois qubits e dois modos harmônicos truncados com N=3, totalizando dimensão 36, com evolução unitária do sistema composto. O resultado de referência salvo registra soma de variação positiva da distância de traço de aproximadamente 0,0830 para o par de estados escolhido.

Há também uma execução identificada como `strong_result`, com backflow de aproximadamente 0,2987, mas fidelidade final otimizada de aproximadamente 0,4643, abaixo da fidelidade livre de referência (~0,9779). Portanto, esse resultado não representa uma melhoria de controle. Ele mostra sensibilidade ao protocolo e deve ser lido como uma execução separada.

A métrica de backflow caracteriza memória para o modelo, par de estados e parâmetros definidos. Não estabelece uma nova lei, não isola um efeito exclusivo da TDT e não deve ser confundida com demonstração de vantagem de controle.

**Reprodução:** o script atual grava a variante `strong_result`. O JSON sem o sufixo `strong` é um registro histórico de uma execução anterior; não deve ser tratado como saída idêntica do script atual.

## 5. Comparação com referência Lindblad

**Arquivos:** `tdt_v5_work_and_markov_reference.py` e `tdt_v5_work_and_markov_reference_result.json`.

A comparação coloca um modelo fechado com modos explícitos ao lado de uma referência Lindblad. O próprio arquivo de resultado alerta que os modelos diferem fisicamente, que a taxa foi calibrada para igualar apenas uma pureza final, que SPSA não certifica o mínimo global e que o trabalho assinado não inclui o custo dos atuadores.

Consequentemente, a diferença calculada entre os trabalhos não é uma assinatura válida da TDT. O resultado é útil como diagnóstico metodológico: mostra por que uma comparação de modelos precisa controlar as diferenças físicas antes de atribuir significado à diferença observada.

## Regras para acrescentar novos resultados

1. Identificar o modelo, as hipóteses, parâmetros e unidades.
2. Guardar código, saída bruta e semente usados em conjunto.
3. Separar comparação exploratória de validação independente.
4. Registrar resultados negativos e inconclusivos sem removê-los do histórico.
5. Não usar palavras como 'prova', 'lei', 'universal' ou 'superior' sem um critério e evidência que sustentem exatamente essa afirmação.
6. Fazer análise dimensional de cada equação antes de compará-la a trabalho, energia livre, calor ou entropia.