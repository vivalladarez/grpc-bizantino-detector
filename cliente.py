import os
os.environ.setdefault("RAY_DISABLE_WINDOWS_JOB_OBJECTS", "1")  # ajuda o Ray no Windows

import grpc
import numpy as np
import ray
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

import bizantinos_pb2 as pb2
import bizantinos_pb2_grpc as pb2_grpc

# -------------------- Parametros --------------------
N_CLIENTES = 5

# Exemplos de cenários:
# 1 cliente malicioso: ["cliente_normal", "cliente_normal", "cliente_normal", "cliente_normal", "cliente_flipa"]
# 2 maliciosos       : ["cliente_normal", "cliente_normal", "cliente_normal", "cliente_altera", "cliente_flipa"]
# 3 maliciosos       : ["cliente_normal", "cliente_normal", "cliente_altera_1feature", "cliente_altera", "cliente_flipa"]
CENARIO = ["cliente_normal", "cliente_normal", "cliente_normal", "cliente_normal", "cliente_flipa"]

# -------------------- Dados Iris --------------------
def particionar_iris_em_clientes(n_clientes=5, seed=42):
    """Particiona o IRIS em n_clientes folds estratificados (sem sobreposição)."""
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    skf = StratifiedKFold(n_splits=n_clientes, shuffle=True, random_state=seed)
    folds = []
    for _, idx in skf.split(X, y):
        folds.append((X[idx], y[idx]))
    return folds

# ------------------- Tipo de Cliente -----------------
def clientes(X, y, rotulo_cliente, seed=None):
    rng = np.random.default_rng(seed)
    X = X.copy()
    y = y.copy()
    n, d = X.shape

    if rotulo_cliente == "cliente_normal":
        return X, y

    if rotulo_cliente == "cliente_altera":
        ruido = rng.normal(loc=0.0, scale=3.0, size=X.shape) # ruído gaussiano em todas as features
        return X + ruido, y

    if rotulo_cliente == "cliente_altera_1feature":
        # escolhe 1 feature: zera ~99% e coloca um grande para deslocar a média
        j = rng.integers(0, d)
        p_zero = 0.99
        mask_zero = rng.random(n) < p_zero
        if mask_zero.all():  # garante ao menos 1 não-zerado
            mask_zero[rng.integers(0, n)] = False

        X[mask_zero, j] = 0.0
        BIG = 1e6  # explodee
        X[~mask_zero, j] = BIG
        return X, y

    if rotulo_cliente == "cliente_flipa":
        return X, (y + 1) % len(np.unique(y))  # flipa rótulos

    return X, y

# -------------- Clientes como Actors ---------------
@ray.remote
class AtorCliente:
    def __init__(self, id_cliente, X, y, rotulo_cliente):
        self.id_cliente = int(id_cliente)
        self.rotulo_cliente = str(rotulo_cliente)
        self.X = X
        self.y = y
        self.canal = grpc.insecure_channel("127.0.0.1:50051")
        self.stub = pb2_grpc.TrainerStub(self.canal)

    def enviar(self):
        grpc.channel_ready_future(self.canal).result(timeout=10)

        seed = 1000 + self.id_cliente
        Xb, yb = clientes(self.X, self.y, self.rotulo_cliente, seed=seed)

        amostras = [
            pb2.Sample(features=list(map(float, x)), label=int(y))
            for x, y in zip(Xb, yb)
        ]
        req = pb2.ClientBatch(client_id=self.id_cliente, samples=amostras)
        reply = self.stub.Submit(req)

        return {
            "id_cliente": self.id_cliente,
            "rotulo_cliente": self.rotulo_cliente,
            "resposta_servidor": reply.detail,
        }

# -------------------- ORQUESTRAÇÃO --------------------
def main():
    ray.init(ignore_reinit_error=True, include_dashboard=False, local_mode=True, num_cpus=N_CLIENTES)

    folds = particionar_iris_em_clientes(N_CLIENTES, seed=42)

    # Completa a lista de cenários para N_CLIENTES com "cliente_normal"
    comportamentos = (CENARIO + ["cliente_normal"] * N_CLIENTES)[:N_CLIENTES]

    # Mapa de comportamentos por id_cliente
    print("\n[CLIENTE] Mapa de comportamentos:")
    for cid, comp in enumerate(comportamentos):
        print(f"  - id_cliente={cid}: {comp}")

    # Cria e dispara atores
    atores = []
    for cid in range(N_CLIENTES):
        Xc, yc = folds[cid]
        rotulo_cliente = comportamentos[cid]
        atores.append(AtorCliente.remote(cid, Xc, yc, rotulo_cliente))

    # Envia lotes para o servidor (em paralelo)
    resultados = ray.get([a.enviar.remote() for a in atores])

    print("\n[CLIENTE] Envios concluídos:")
    for r in sorted(resultados, key=lambda z: z["id_cliente"]):
        print(f"  - id_cliente={r['id_cliente']:>2} | comportamento={r['rotulo_cliente']:<20} | {r['resposta_servidor']}")

    # Solicita treino/relatório ao servidor (esperando N_CLIENTES)
    canal = grpc.insecure_channel("127.0.0.1:50051")
    grpc.channel_ready_future(canal).result(timeout=10)
    stub = pb2_grpc.TrainerStub(canal)
    resp = stub.Train(pb2.TrainRequest(expected_clients=N_CLIENTES))

    # Resultado final (com totais e lista de bizantinos)
    ids_biz = list(resp.byzantine_clients)
    num_biz = len(ids_biz)
    total = resp.total_clients

    print("\n[CLIENTE] RESULTADOS")
    print(f"  total_clientes        : {total}")
    print(f"  bizantinos_detectados : {num_biz} -> {ids_biz if num_biz else '[]'}")
    for cid in ids_biz:
        if 0 <= cid < len(comportamentos):
            print(f"    - id={cid} | comportamento={comportamentos[cid]}")

    print(f"  train_acc_all         : {resp.train_acc_all:.4f}")
    print(f"  test_acc_all          : {resp.test_acc_all:.4f}")
    print(f"  train_acc_filtrado    : {resp.train_acc_filtered:.4f}")
    print(f"  test_acc_filtrado     : {resp.test_acc_filtered:.4f}")

    ray.shutdown()

if __name__ == "__main__":
    main()
