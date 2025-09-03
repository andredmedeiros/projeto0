import cv2
import numpy as np

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Horizontal projection
    projection = np.sum(edges, axis=0)
    peaks = np.where((projection[1:-1] > projection[:-2]) & (projection[1:-1] > projection[2:]))[0] + 1

    # Calcular comprimentos de onda
    wavelengths = []
    if len(peaks) > 1:
        wavelengths = np.diff(peaks)
        avg_wavelength = int(np.mean(wavelengths))
    else:
        avg_wavelength = 0

    # Gráfico mais bonito
    graph_height = 200
    width = frame.shape[1]
    graph = np.ones((graph_height, width, 3), dtype=np.uint8) * 255  # fundo branco

    # Normalizar projeção
    norm_proj = (projection / (projection.max() + 1e-5) * (graph_height - 40)).astype(np.int32)
    # Desenhar curva suavizada
    pts = np.array([[x, graph_height-20 - y] for x, y in enumerate(norm_proj)], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(graph, [pts], False, (50, 100, 220), 2, cv2.LINE_AA)

    # Eixos
    cv2.line(graph, (0, graph_height-20), (width, graph_height-20), (0,0,0), 1)
    cv2.line(graph, (40, 0), (40, graph_height), (0,0,0), 1)
    cv2.putText(graph, "Projecao Horizontal", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(graph, f"Media lambda: {avg_wavelength}", (width-220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # Desenhar picos e comprimentos de onda
    for x in peaks:
        cv2.line(graph, (x, 0), (x, graph_height-20), (0, 180, 255), 1)
        cv2.circle(graph, (x, graph_height-20 - norm_proj[x]), 4, (0, 0, 255), -1)
    for i in range(len(peaks)-1):
        x1, x2 = peaks[i], peaks[i+1]
        y = graph_height-30
        cv2.arrowedLine(graph, (x1, y), (x2, y), (100, 100, 255), 2, tipLength=0.05)
        wl = x2 - x1
        cv2.putText(graph, str(wl), ((x1+x2)//2-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,0,200), 2)

    # Adicionar legenda
    cv2.rectangle(graph, (10, graph_height-18), (180, graph_height-2), (255,255,255), -1)
    cv2.putText(graph, "Linha: Projecao", (15, graph_height-5), cv2.FONT_HERSHEY_PLAIN, 1, (50,100,220), 1)
    cv2.putText(graph, "Picos", (120, graph_height-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

    # Redimensionar frame para caber acima do gráfico
    frame_resized = cv2.resize(frame, (width, width*frame.shape[0]//frame.shape[1]))
    combined = np.vstack((frame_resized, graph))
    return combined

def capture_and_show():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Can't receive frame.")
        return

    processed = process_frame(frame)
    cv2.imshow('Imagem Capturada + Grafico', processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_show()
