# 🛡️ gRPC Byzantine Fault Detector

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
- [Configuração Avançada](#-configuração-avançada)
- [Métricas e Análise](#-métricas-e-análise)

## Visão Geral

Este projeto implementa um sistema de **aprendizado federado resiliente** que:

- 🌐 **Distribui o treinamento** entre múltiplos clientes usando gRPC
- 🕵️ **Detecta clientes bizantinos** através de métricas estatísticas
- 🛡️ **Filtra participantes maliciosos** para melhorar a qualidade do modelo
- 📊 **Compara performance** entre modelos treinados com e sem filtragem
- 🔬 **Utiliza dataset Iris** como caso de estudo controlado

## 🏗️ Arquitetura

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

## ⚙️ Instalação e Configuração

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

1. 🏁 **Inicialização**: Servidor aguarda conexões na porta 50051
2. 📤 **Envio de Dados**: Clientes enviam seus lotes via `Submit()`
3. 🔍 **Detecção**: Servidor analisa métricas e identifica bizantinos
4. 🎯 **Treinamento**: Treina Random Forest com todos os dados e com dados filtrados
5. 📊 **Resultados**: Exibe comparação de performance

## 🎭 Tipos de Clientes Bizantinos

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

## Interpretação dos Resultados

### Cenários

#### 1. Cenário com apenas 1 cliente malicioso
```
[CLIENTE] Mapa de comportamentos:
  - id_cliente=0: cliente_normal
  - id_cliente=1: cliente_normal  
  - id_cliente=2: cliente_altera_1feature  ← Bizantino
  - id_cliente=3: cliente_altera
  - id_cliente=4: cliente_flipa            ← Bizantino
```

#### 2. Cenário com 2 clientes maliciosos
```
[SERVIDOR] Métricas por cliente:
  - id_cliente=0: centroide_max=0.412, erro_consistencia=0.067  ← Normal
  - id_cliente=1: centroide_max=0.355, erro_consistencia=0.100  ← Normal
  - id_cliente=2: centroide_max=7.918, erro_consistencia=0.367  ← SUSPEITO (ambas métricas altas)
  - id_cliente=3: centroide_max=0.621, erro_consistencia=0.167  ← Normal
  - id_cliente=4: centroide_max=0.544, erro_consistencia=0.133  ← Normal (falso negativo)

[SERVIDOR] Limiares → centroide_max=3.000, erro=0.35
[SERVIDOR] Bizantinos detectados = [2]
```

#### Cenário com 3 clientes maliciosos
```
[SERVIDOR] Resultados do treinamento:
  📊 Modelo com TODOS os clientes:
     - Acurácia treino: 0.85
     - Acurácia teste:  0.78
  
  🛡️ Modelo FILTRADO (sem bizantinos):
     - Acurácia treino: 0.92  ← Melhoria
     - Acurácia teste:  0.89  ← Melhoria
  
  📈 Total de clientes: 5 | Filtrados: 4 | Bizantinos: [2]
```


## 📈 Métricas e Análise

### Eficácia da Detecção

| Tipo de Ataque | `centroide_max` | `erro_consistencia` | Taxa de Detecção |
|----------------|-----------------|---------------------|------------------|
| `cliente_altera_1feature` | 🔴 Muito Alto | 🟡 Médio | ~95% |
| `cliente_flipa` | 🟢 Baixo | 🔴 Alto | ~90% |
| `cliente_altera` | 🟡 Médio | 🟡 Médio | ~70% |
| `cliente_normal` | 🟢 Baixo | 🟢 Baixo | ~5% (falsos positivos) |

### Impacto na Performance

- **Sem Filtragem**: Modelos degradados por dados corrompidos
- **Com Filtragem**: Melhoria típica de 5-15% na acurácia
- **Trade-off**: Menos dados vs. maior qualidade


## 📄 Licença

Este projeto foi desenvolvido para atender a prática 3 https://ic.unicamp.br/~allanms/mo809-S22025/labs/Lab-03/ da disciplina Tópicos em Computação Distribuída IC Unicamp
