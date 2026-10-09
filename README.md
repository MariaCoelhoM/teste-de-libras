# Libras - reconhecimento de alfabeto (estatico) e palavras (dinamico)

Projeto de TCC: reconhecimento de sinais de Libras a partir de landmarks de
maos, rosto (e, no alfabeto, so maos) extraidos com MediaPipe, classificados
por uma CNN (alfabeto) e uma LSTM (palavras).

## Estrutura

```
libras/                                            modulo comum (NAO duplicar funcoes daqui em outro lugar)
  landmarks.py                                      detectores MediaPipe, normalizacao, amostragem de frames, leitura de zip/CSV
  training_utils.py                                 filtro --classes, sinalizador, contagem de amostras

extract_landmarks_imagem.py                          extrai landmarks do alfabeto (imagens, dataset archive/train e archive/test)
extract_landmarks_video_vlibras.py                   extrai landmarks so-maos de videos (V-Librasil)
extract_landmarks_video_hands_face_minds_libras.py   extrai landmarks maos+rosto de videos (MINDS-Libras)

train_cnn.py                                         treina o modelo do alfabeto (CNN 1D)
train_lstm.py                                        treina o modelo de palavras so-maos (LSTM)
train_lstm_hands_face.py                             treina o modelo de palavras maos+rosto (LSTM)
matriz.py                                             gera a matriz de confusao de um modelo ja treinado

app.py                                                servidor Flask (demo ao vivo pela webcam)
templates/index.html                                  pagina da demo (modo estatico e dinamico)
```

## 1. Instalar dependencias

```bash
pip install uv
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version   # deve mostrar 3.11.x
uv pip install -r requirements.txt
uv pip uninstall opencv-python opencv-contrib-python
uv pip install opencv-python-headless==4.10.0.84
```

As versoes em `requirements.txt` sao um ponto de partida - se o `pip install`
der conflito no seu ambiente, ajuste a versao do pacote que falhar e teste de
novo (nao tenho como validar contra o seu Python/SO exatos daqui).

## 2. Baixar os modelos do MediaPipe

```bash
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

wget -O face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Ambos ficam na raiz do projeto (mesma pasta do `app.py`).

## 3. Datasets

### Alfabeto (imagens)

Pasta `archive/`, com a estrutura original do dataset preservada:

```
archive/
  train/<letra>/*.jpg
  test/<letra>/*.jpg
```

<!-- TODO: preencher com a fonte exata do dataset do alfabeto (nome, link,
licenca) e confirmar se ele fica versionado no repositorio ou precisa ser
baixado separadamente. -->

### MINDS-Libras (palavras, video)

<!-- TODO: preencher com o link/fonte oficial do MINDS-Libras e a licenca de uso. -->

Extraido com `extract_landmarks_video_hands_face_minds_libras.py` (maos +
rosto), usando o nome do arquivo pra identificar classe e sinalizador (ex.:
`01AcontecerSinalizador02-5.mp4` -> classe `Acontecer`, sinalizador `02`).

### V-Librasil (palavras, video)

<!-- TODO: preencher com o link/fonte oficial do V-Librasil, a licenca, e
como o annotations.csv e estruturado (colunas video_name/class). -->

Extraido com `extract_landmarks_video_vlibras.py` (so maos), via
`--annotations_csv annotations.csv` ou `--dataset_dir` com uma subpasta por
classe.

## 4. Extrair landmarks

Alfabeto:

```bash
python extract_landmarks_imagem.py --dataset_dir ./archive --output landmarks.npz
```

MINDS-Libras:

```bash
python extract_landmarks_video_hands_face_minds_libras.py \
  --dataset_dir ./minds-libras \
  --filename_regex '^\d+([A-Za-zÀ-ÿ]+)Sinalizador\d+-\d+\.mp4$' \
  --output landmarks_video_minds.npz \
  --failures_log falhas_minds.csv \
  --quiet
```

V-Librasil:

```bash
python extract_landmarks_video_vlibras.py \
  --dataset_dir ./v-librasil \
  --annotations_csv annotations.csv \
  --output landmarks_video_vlibrasil.npz
```

## 5. Treinar

Alfabeto (usa a divisao `train/`/`test/` original do dataset automaticamente,
se o `.npz` tiver essa informacao):

```bash
python train_cnn.py --data landmarks.npz --output modelo_alfabeto.keras \
  --counts_output resultados_alfabeto_contagem.csv
```

Palavras, MINDS-Libras (maos + rosto), split por sinalizador:

```bash
python train_lstm_hands_face.py --data landmarks_video_minds.npz \
  --output modelo_palavras_minds.keras \
  --counts_output resultados_minds_contagem.csv
```

Pra comparar com/sem rosto, repita com `--no_face`.

Palavras, V-Librasil (so maos), deixando um sinalizador de fora por rodada
(poucos sinalizadores - nao da pra fazer um split fixo de treino/teste):

```bash
python train_lstm.py --data landmarks_video_vlibrasil.npz \
  --leave_one_signer_out \
  --signer_regex "SEU_PADRAO_AQUI" \
  --counts_output resultados_vlibrasil_contagem.csv
```

<!-- TODO: confirmar o padrao de nome de arquivo do V-Librasil pra preencher
--signer_regex corretamente (o padrao default, Sinalizador(\d+), e do
MINDS-Libras). -->

Pra treinar so um subconjunto do vocabulario, use `--classes` em qualquer um
dos tres treinos: `--classes "Aluno,Filho,Medo"` ou `--classes @lista.txt`
(uma classe por linha).

## 6. Avaliar (matriz de confusao)

```bash
python matriz.py --data landmarks_video_minds.npz --model modelo_palavras_minds.keras
```

Reproduz o mesmo split por sinalizador usado no treino (precisa dos mesmos
`--test_size`/`--signer_regex`, se voce mudou o padrao).

## 7. Rodar a demo

```bash
python app.py
```

Abre em `http://localhost:5000` (ou a porta exposta pelo Codespace). A pagina
tem um seletor entre modo "Alfabeto (estatico)" e "Palavras (dinamico)".

## Observacoes

- As funcoes de deteccao, normalizacao e leitura de zip/CSV moram em
  `libras/` e sao compartilhadas por todos os extratores - no reimplemente
  localmente num script novo, importe de la.
- `extract_landmarks_video_hands_face_minds_libras.py` e
  `extract_landmarks_video_vlibras.py` ainda tem cada um sua propria
  `find_class_dirs` de video (`find_class_dirs_video` em `libras/landmarks.py`,
  compartilhada entre os dois) - e diferente do `find_class_dirs` do alfabeto
  (que rastreia o split `train`/`test`), por isso nao foi unificada com ela.
