# Connect4 Reinforcement Learning Agent

Proyecto de **aprendizaje por refuerzo** aplicado al juego **Connect4**, desarrollado para entrenar un agente capaz de jugar de forma competitiva contra distintos oponentes predefinidos.

El modelo principal está basado en **Deep Q-Learning (DQN)** y fue entrenado de manera iterativa contra agentes como `RandomAgent`, `DefenderAgent` y versiones heurísticas, buscando mejorar su desempeño de forma progresiva. :contentReference[oaicite:2]{index=2}

## Objetivo

Entrenar un agente que aprenda a jugar Connect4 y logre un buen rendimiento frente a oponentes de distinta dificultad.

## Archivos principales

- `connect4.py`: lógica del juego
- `agentes.py`: definición de agentes y estrategias
- `entrenar.py`: entrenamiento del agente
- `evaluar.py`: evaluación del modelo entrenado
- `jugar_humano_contra_defensor.py`: partida entre humano y agente defensor
- `principal.py`: punto de entrada principal
- `utils.py` / `utils_testing.py`: funciones auxiliares
- `trained_agent_model.pth`: modelo entrenado
- `Informe - TP2.pdf`: informe del trabajo

## Resultados

Después de múltiples pruebas y ajustes, se obtuvo un agente que:

- gana la mayoría de las partidas contra `RandomAgent`
- muestra un comportamiento más ordenado y consistente
- compite razonablemente bien contra oponentes más complejos
- todavía tiene dificultades frente a `DefenderAgent`, que sigue siendo un rival desafiante :contentReference[oaicite:3]{index=3}

## Hiperparámetros elegidos

- `gamma = 0.99`
- `epsilon_start = 0.12`
- `epsilon_min = 0.05`
- `epsilon_decay = 0.997`
- `learning_rate = 0.00005`
- `batch_size = 512`
- `memory_size = 200000`
- `target_update_every = 1000` :contentReference[oaicite:4]{index=4}

## Cómo correrlo

Ejemplo general:

```bash
python principal.py
