# servidor.py — detecção mínima (centróide + consistência de rótulos)
from concurrent import futures
import grpc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import bizantinos_pb2 as pb2
import bizantinos_pb2_grpc as pb2_grpc
# Se estiver usando trainer.proto, troque as duas linhas acima por:
# import trainer_pb2 as pb2
# import trainer_pb2_grpc as pb2_grpc

# -------------------- ARMAZENAMENTO --------------------
ARMAZENAMENTO = {}  # id_cliente -> (X, y)

# -------------------- DETECÇÃO BIZANTINA (mínima, 2 sinais) --------------------
def logica_bizantina(por_cliente, limiar_erro=0.35, iqr_k=3.0):
    """
    Detector mínimo:
      - centroide_max: maior desvio (|mu_cliente_feature - mu_global_feature|) em z-score global.
      - erro_consistencia: erro de nearest-centroid (consistência de rótulos).
    Suspeito se centroide_max > (mediana + iqr_k*IQR) ou erro_consistencia > limiar_erro.
    """
    clientes = sorted(por_cliente.keys())
    X_todos = np.vstack([por_cliente[c][0] for c in clientes])
    y_todos = np.concatenate([por_cliente[c][1] for c in clientes])
    classes = np.unique(y_todos)

    # Padronização global (z-score)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_todos)

    # offsets por cliente
    quantidades = [len(por_cliente[c][1]) for c in clientes]
    offsets = np.cumsum([0] + quantidades)

    # média global por feature (em z)
    mu_global = Xz.mean(axis=0)

    # centróides globais por classe (em z) para nearest-centroid
    MU = np.stack([Xz[y_todos == k].mean(axis=0) for k in classes], axis=0)  # (K, D)

    centroide_max = {}
    erro_consistencia = {}

    for i, c in enumerate(clientes):
        s, e = offsets[i], offsets[i + 1]
        Xc = Xz[s:e]
        yc = y_todos[s:e]

        # Maior desvio da média do cliente em relação à média global (por feature)
        centroide_max[c] = float(np.abs(Xc.mean(axis=0) - mu_global).max())

        # Erro de consistência via nearest-centroid
        dists = np.linalg.norm(Xc[:, None, :] - MU[None, :, :], axis=2)  # (n_c, K)
        pred = classes[np.argmin(dists, axis=1)]
        erro_consistencia[c] = float((pred != yc).mean())

    # Limiar robusto para centroide_max: mediana + iqr_k * IQR
    valores = np.array(list(centroide_max.values()), dtype=float)
    q1, q3 = np.percentile(valores, [25, 75])
    iqr = q3 - q1
    limiar_centroide_max = float(np.median(valores) + (iqr_k * iqr if iqr > 0 else 3.0))

    suspeitos = {
        c for c in clientes
        if (centroide_max[c] > limiar_centroide_max) or (erro_consistencia[c] > limiar_erro)
    }

    metricas = {
        "centroide_max": centroide_max,
        "erro_consistencia": erro_consistencia,
        "limiar_centroide_max": limiar_centroide_max,
        "limiar_erro": limiar_erro,
    }
    return suspeitos, metricas

# -------------------- SERVIÇO --------------------
class TreinadorServicer(pb2_grpc.TrainerServicer):
    def Submit(self, request, context):
        X = np.array([list(s.features) for s in request.samples], dtype=np.float32)
        y = np.array([s.label for s in request.samples], dtype=np.int64)
        ARMAZENAMENTO[int(request.client_id)] = (X, y)
        detalhe = (
            f"Recebido id_cliente={request.client_id} | amostras={len(y)} "
            f"| total_clientes={len(ARMAZENAMENTO)}"
        )
        print("[SERVIDOR]", detalhe)
        return pb2.SubmitReply(ok=True, detail=detalhe, received_clients=len(ARMAZENAMENTO))

    def Train(self, request, context):
        esperados = request.expected_clients if request.expected_clients > 0 else 5
        if len(ARMAZENAMENTO) < esperados:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Aguardando {esperados} clientes; recebidos {len(ARMAZENAMENTO)}.")
            return pb2.TrainResponse()

        clientes = sorted(ARMAZENAMENTO.keys())
        X_todos = np.vstack([ARMAZENAMENTO[c][0] for c in clientes])
        y_todos = np.concatenate([ARMAZENAMENTO[c][1] for c in clientes])

        # --- Detecção mínima ---
        suspeitos, M = logica_bizantina(ARMAZENAMENTO, limiar_erro=0.35, iqr_k=3.0)
        biz = sorted(list(suspeitos))
        mantidos = [c for c in clientes if c not in suspeitos]

        # Logs compactos
        print("[SERVIDOR] Métricas por cliente:")
        for c in clientes:
            print(
                f"  - id_cliente={c}: centroide_max={M['centroide_max'][c]:.3f}, "
                f"erro_consistencia={M['erro_consistencia'][c]:.3f}"
            )
        print(
            f"[SERVIDOR] Limiares -> centroide_max={M['limiar_centroide_max']:.3f}, "
            f"erro={M['limiar_erro']:.2f}"
        )
        print(f"[SERVIDOR] Bizantinos detectados = {biz}")

        # --- Treino/avaliação com TODOS ---
        Xtr, Xte, ytr, yte = train_test_split(
            X_todos, y_todos, test_size=0.2, random_state=42, stratify=y_todos
        )
        clf_todos = RandomForestClassifier(n_estimators=300, random_state=42)
        clf_todos.fit(Xtr, ytr)
        acc_treino_todos = float(clf_todos.score(Xtr, ytr))
        acc_teste_todos  = float(clf_todos.score(Xte, yte))

        # --- Treino/avaliação SEM bizantinos ---
        if mantidos:
            Xk = np.vstack([ARMAZENAMENTO[c][0] for c in mantidos])
            yk = np.concatenate([ARMAZENAMENTO[c][1] for c in mantidos])
            Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(
                Xk, yk, test_size=0.2, random_state=42, stratify=yk
            )
            clf_filtrado = RandomForestClassifier(n_estimators=300, random_state=42)
            clf_filtrado.fit(Xtr_f, ytr_f)
            acc_treino_filtrado = float(clf_filtrado.score(Xtr_f, ytr_f))
            acc_teste_filtrado  = float(clf_filtrado.score(Xte_f, yte_f))
        else:
            acc_treino_filtrado = 0.0
            acc_teste_filtrado  = 0.0

        return pb2.TrainResponse(
            train_acc_all=acc_treino_todos,
            train_acc_filtered=acc_treino_filtrado,
            test_acc_all=acc_teste_todos,
            test_acc_filtered=acc_teste_filtrado,
            total_clients=len(clientes),
            filtered_clients=len(biz),
            byzantine_clients=biz
        )

# -------------------- BOOT --------------------
def serve():
    servidor = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb2_grpc.add_TrainerServicer_to_server(TreinadorServicer(), servidor)
    bind_addr = "0.0.0.0:50051"  # ou "127.0.0.1:50051" se for apenas local
    if servidor.add_insecure_port(bind_addr) <= 0:
        raise RuntimeError(f"Falha ao bindar em {bind_addr}")
    servidor.start()
    print(f"[SERVIDOR] gRPC escutando em {bind_addr}")
    servidor.wait_for_termination()

if __name__ == "__main__":
    serve()
