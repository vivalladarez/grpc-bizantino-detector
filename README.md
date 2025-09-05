# gRPC Byzantine Fault Detector

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol-green.svg)](https://grpc.io/)
[![Ray](https://img.shields.io/badge/Ray-Distributed-orange.svg)](https://ray.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red.svg)](https://scikit-learn.org/)

Sistema de treinamento distribuído com **gRPC** e **Ray** que simula clientes bizantinos (maliciosos) e implementa algoritmos de detecção para identificar e filtrar participantes suspeitos durante o treinamento federado.

## Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#️-arquitetura)
- [Instalação e Configuração](#️-instalação-e-configuração)
- [Como Executar](#-como-executar)
- [Tipos de Clientes Bizantinos](#-tipos-de-clientes-bizantinos)
- [Algoritmo de Detecção](#-algoritmo-de-detecção)
- [Interpretação dos Resultados](#-interpretação-dos-resultados)
- [Métricas e Análise](#-métricas-e-análise)

## Visão Geral

Este projeto implementa um sistema de **aprendizado federado resiliente** que:

- **Distribui o treinamento** entre múltiplos clientes usando gRPC
- **Detecta clientes bizantinos** através de métricas estatísticas
- **Filtra participantes maliciosos** para melhorar a qualidade do modelo
- **Compara performance** entre modelos treinados com e sem filtragem
- **Utiliza dataset Iris** como caso de estudo controlado

## Arquitetura

```
Cliente 1 (Normal)    ─┐
Cliente 2 (Normal)    ─┤
Cliente 3 (Bizantino) ─┼─► Servidor gRPC ─► Detector Bizantino ─► Random Forest
Cliente 4 (Bizantino) ─┤                                          ├─► Modelo Completo
Cliente 5 (Normal)    ─┘                                          └─► Modelo Filtrado
```

### Componentes Principais

- **`servidor.py`**: Servidor gRPC que recebe dados, detecta bizantinos e treina modelos
- **`cliente.py`**: Simulador de clientes com diferentes comportamentos (normais e bizantinos)
- **`bizantinos.proto`**: Definição do protocolo gRPC para comunicação

## Instalação e Configuração

### Pré-requisitos

- **Python 3.9+**
- **pip** (gerenciador de pacotes)

### 1. Instalar Dependências

```bash
pip install grpcio grpcio-tools ray scikit-learn numpy
```

### 2. Gerar Stubs gRPC

```bash
python -m grpc_tools.protoc -I . \
  --python_out . \
  --grpc_python_out . \
  bizantinos.proto
```

Isso criará os arquivos:
- `bizantinos_pb2.py` (mensagens)
- `bizantinos_pb2_grpc.py` (serviços)

## Como Executar

### Execução Básica

1. **Inicie o servidor** (Terminal 1):
```bash
python servidor.py
```

2. **Execute os clientes** (Terminal 2):
```bash
python cliente.py
```

### Fluxo de Execução

1. **Inicialização**: Servidor aguarda conexões na porta 50051
2. **Envio de Dados**: Clientes enviam seus lotes via `Submit()`
3. **Detecção**: Servidor analisa métricas e identifica bizantinos
4. **Treinamento**: Treina Random Forest com todos os dados e com dados filtrados
5. **Resultados**: Exibe comparação de performance

## Tipos de Clientes Bizantinos

### 1. `cliente_normal`
**Comportamento**: Envia dados íntegros sem modificações
- Dados originais do dataset Iris
- Labels corretos
- Features inalteradas

### 2. `cliente_altera`
**Comportamento**: Adiciona ruído gaussiano em todas as features
- Ruído: `N(μ=0, σ=3.0)`
- Aumenta variância dos dados
- Mantém labels originais

### 3. `cliente_altera_1feature`
**Comportamento**: Ataque direcionado em uma feature específica
- Escolhe 1 feature aleatoriamente
- Zera ~99% dos valores dessa feature
- Coloca valores extremos nos restantes (~50x a média)
- **Mais detectável** pelo `centroide_max`

### 4. `cliente_flipa`
**Comportamento**: Corrompe labels ciclicamente
- Label 0 → 1, Label 1 → 2, Label 2 → 0
- **Mais detectável** pelo `erro_consistencia`
- Mantém features inalteradas

## Algoritmo de Detecção

O sistema implementa um **detector híbrido** baseado em duas métricas complementares:

### Métricas de Detecção

#### 1. `centroide_max`
```
centroide_max = max|μ_cliente_feature - μ_global_feature| (em z-score)
```
- **Detecta**: Ataques em features (ex: `cliente_altera_1feature`)
- **Limiar**: Mediana + 3.0 × IQR dos centróides máximos
- **Princípio**: Clientes bizantinos têm distribuições muito diferentes

#### 2. `erro_consistencia`
```
erro_consistencia = erro_nearest_centroid(dados_cliente, centróides_globais)
```
- **Detecta**: Corrupção de labels (ex: `cliente_flipa`)
- **Limiar**: 0.35 (configurável)
- **Princípio**: Labels corrompidos não se alinham com centróides globais

### Critério de Detecção

Um cliente é considerado **bizantino** se:
```
centroide_max > (mediana + 3.0×IQR) OU erro_consistencia > 0.35
```

## Resultados

### Cenários

#### 1. Cenário com apenas 1 cliente malicioso

```
CENARIO = ["cliente_normal", "cliente_normal", "cliente_normal", "cliente_normal", "cliente_flipa"]
```

```
[SERVIDOR] Métricas por cliente:
  - id_cliente=0: centroide_max=0.206, erro_consistencia=0.067
  - id_cliente=1: centroide_max=0.174, erro_consistencia=0.133
  - id_cliente=2: centroide_max=0.071, erro_consistencia=0.167
  - id_cliente=3: centroide_max=0.053, erro_consistencia=0.033
  - id_cliente=4: centroide_max=0.094, erro_consistencia=0.867

--------------------------------------------------------------

[CLIENTE] RESULTADOS
  total_clientes        : 5
  bizantinos_detectados : 1 -> [4]
    - id=4 | comportamento=cliente_flipa
  train_acc_all         : 0.9917
  test_acc_all          : 0.6667
  train_acc_filtrado    : 1.0000
  test_acc_filtrado     : 0.9583
```

#### 2. Cenário com 2 clientes maliciosos

```
CENARIO = ["cliente_normal", "cliente_normal", "cliente_normal", "cliente_altera", "cliente_flipa"]
```

```
[SERVIDOR] Métricas por cliente:
  - id_cliente=0: centroide_max=0.212, erro_consistencia=0.167
  - id_cliente=1: centroide_max=0.078, erro_consistencia=0.167
  - id_cliente=2: centroide_max=0.109, erro_consistencia=0.167
  - id_cliente=3: centroide_max=0.497, erro_consistencia=0.367
  - id_cliente=4: centroide_max=0.143, erro_consistencia=0.833
[SERVIDOR] Limiares -> centroide_max=0.452, erro=0.35
[SERVIDOR] Bizantinos detectados = [3, 4]

--------------------------------------------------------------

[CLIENTE] Resultados
  total_clientes        : 5
  bizantinos_detectados : 2 -> [3, 4]
    - id=3 | comportamento=cliente_altera
    - id=4 | comportamento=cliente_flipa
  train_acc_all         : 1.0000
  test_acc_all          : 0.6000
  train_acc_filtrado    : 1.0000
  test_acc_filtrado     : 0.9444
```

#### Cenário com 3 clientes maliciosos

```
CENARIO = ["cliente_normal", "cliente_normal", "cliente_altera_1feature", "cliente_altera", "cliente_flipa"]
```

```
[SERVIDOR] Métricas por cliente:
  - id_cliente=0: centroide_max=0.362, erro_consistencia=0.200
  - id_cliente=1: centroide_max=0.291, erro_consistencia=0.167
  - id_cliente=2: centroide_max=1.072, erro_consistencia=0.400
  - id_cliente=3: centroide_max=0.497, erro_consistencia=0.367
  - id_cliente=4: centroide_max=0.324, erro_consistencia=0.833
[SERVIDOR] Limiares -> centroide_max=0.881, erro=0.35
[SERVIDOR] Bizantinos detectados = [2, 3, 4]

--------------------------------------------------------------

[CLIENTE] Resultados
  total_clientes        : 5
  bizantinos_detectados : 3 -> [2, 3, 4]
    - id=2 | comportamento=cliente_altera_1feature
    - id=3 | comportamento=cliente_altera
    - id=4 | comportamento=cliente_flipa
  train_acc_all         : 1.0000
  test_acc_all          : 0.5333
  train_acc_filtrado    : 1.0000
  test_acc_filtrado     : 1.0000
```

## 📄 Licença

Este projeto foi desenvolvido para atender a prática 3 https://ic.unicamp.br/~allanms/mo809-S22025/labs/Lab-03/ da disciplina Tópicos em Computação Distribuída IC Unicamp

[![Laboratório 2]](https://ic.unicamp.br/~allanms/mo809-S22025/labs/Lab-02/)
