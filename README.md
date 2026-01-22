# Desafio Participa DF — Categoria Acesso à Informação

## Detector Híbrido de Dados Pessoais em Pedidos e-SIC

Solução desenvolvida para o **1º Hackathon em Controle Social – Desafio Participa DF**, com foco na **identificação automática de pedidos de acesso à informação que contenham dados pessoais**, apoiando a correta aplicação da LGPD e a triagem eficiente de solicitações.

---

## Objetivo
Detectar automaticamente pedidos que contenham **dados pessoais identificáveis**, como:
- CPF
- Telefone
- E-mail
- RG e identificadores numéricos
- Combinações que permitam identificar uma pessoa natural

A solução prioriza **redução de falsos negativos**, explicabilidade e reprodutibilidade.

---

## Conformidade com LGPD
- Os dados utilizados são **amostrais e descaracterizados**, fornecidos exclusivamente para o Desafio Participa DF.
- O dataset **não é versionado no repositório**.
- É vedada qualquer tentativa de reidentificação, reconstrução de registros ou uso fora do escopo do desafio, conforme a **Lei nº 13.709/2018 (LGPD)**.

---

## Abordagem Técnica

A solução adota uma **arquitetura híbrida explicável**, composta por três camadas:

### 1️ Regras (Regex)
Identificação direta de padrões de dados pessoais:
- CPF
- E-mail
- Telefone
- Outros identificadores numéricos relevantes

As regras funcionam como **camada de segurança**, evitando falsos negativos em casos evidentes.

### 2️ Modelo Estatístico (Machine Learning)
- Vetorização **TF-IDF**
- Classificador **Regressão Logística**
- Aprende padrões linguísticos associados a pedidos com dados pessoais
- Não utiliza APIs externas ou modelos proprietários

### 3️ Score Híbrido
Combinação ponderada entre regras e modelo estatístico:

score_final = 0.45 * ml_score + 0.55 * regex_score

**Configuração padrão:**
- alpha = 0.45
- threshold = 0.25
- regex_force_thr = 0.35

Sempre que o regex indica forte evidência de dado pessoal, o pedido é classificado como positivo, mesmo que o score estatístico seja baixo.  
Estratégia intencionalmente orientada à **redução de falsos negativos**.

---

## Estrutura do Projeto

src/
  run.py
  config.py

  models/
    make_synth_dataset.py
    predict.py
    predict_hybrid.py
    report_preds.py
    train.py
    train_hybrid.py

  features/
    regex_features.py

  io/
    load_data.py
    schemas.py

  utils/
    metrics.py

data/
  .gitkeep



---

## Requisitos
- Python **3.10 ou superior**

---

## Instalação

python -m venv .venv

Ativação do ambiente:

Windows:
.venv\Scripts\Activate.ps1

Linux / Mac:
source .venv/bin/activate

Instalação das dependências:
pip install -r requirements.txt

---

## Execução
Basta rodar o comando abaixo, não se esqueça de especificar o caminho do arquivo que será analisado, ele pode ser .xlsx, .csv, .jsonl.

python -m src.run --input CAMINHO_DO_ARQUIVO.xlsx --output artifacts/reports/preds_final.csv --mode auto

### Modos disponíveis
- auto   → Usa o modelo híbrido se existir; caso contrário, executa apenas regras (regex)
- regex  → Apenas regras
- hybrid → Força uso do modelo híbrido (requer modelo treinado localmente)

O modelo híbrido é treinado localmente e **não é versionado no repositório**.

---

## Entrada de Dados
- Formatos suportados: .xlsx, .csv, .jsonl
- A coluna de texto é detectada automaticamente (ex.: Texto Mascarado, texto, mensagem, pedido)

---

## Saída
Arquivo CSV padronizado contendo:
- id
- pred_label (0 = sem dados pessoais, 1 = contém dados pessoais)
- pred_score
- ml_score (quando aplicável)
- regex_score
- has_cpf
- has_email
- has_phone
- forced_by_regex (quando aplicável)

---

## Reprodutibilidade
- Seeds fixas
- Execução determinística
- Sem dependência de serviços externos
- Pipeline completo executa em poucos minutos

---

## Conclusão
A solução entrega um pipeline robusto, transparente e alinhado aos princípios de controle social, LGPD e governança de dados, oferecendo **triagem automatizada confiável** para pedidos de acesso à informação no contexto do Desafio Participa DF.
