# grpc-bizantino-detector

Treinamento distribuído com gRPC e Ray, simulando clientes bizantinos (maliciosos) que corrompem os dados. O servidor recebe os lotes dos clientes, detecta e filtra participantes suspeitos e treina um modelo clássico (Random Forest) com e sem filtragem para comparar impacto.

------------------------

## Requisitos 

Python 3.9+

Dependências:

pip install grpcio grpcio-tools ray scikit-learn numpy

