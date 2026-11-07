# Guia de submissão — IEEE Access

Artigo: *Day-Ahead Load Forecasting Under Rapid Behind-the-Meter Solar Growth:
An Open Benchmark Against a System Operator's Official Forecast in Brazil*

PDF pronto: `submissao/artigo_ieee_access.pdf` (compilado de `paper/main.tex`) (12 páginas, template oficial IEEE Access).

## Antes de submeter (o que só você pode fazer)

1. **Ler o artigo inteiro.** Você é o autor responsável. Confira afiliação,
   nome, e-mail e se concorda com todas as afirmações.
2. **ORCID.** Crie ou confirme seu ORCID (orcid.org) e cadastre o mesmo ORCID
   em "Meus Dados" no portal de periódicos CAPES. O ScholarOne da IEEE exige
   ORCID do autor que submete.
3. **E-mail institucional.** Use `izialber.muniz@aluno.cefet-rj.br` como
   e-mail do autor correspondente no ScholarOne. É por ele que a IEEE
   reconhece o vínculo com o CEFET/RJ e aplica o acordo CAPES (isenção da APC).
4. **Repositório público.** O artigo diz que o código está em
   `github.com/izialber/artigo-ieee`. Antes de submeter:
   - faça o merge do branch `claude/jolly-goldberg-g7ty1v` no `main`;
   - mude o repositório para **público** (Settings → General → Danger Zone →
     Change visibility).
   - Opcional: crie um release e conecte ao Zenodo para ter um DOI do código.
5. **Foto e biografia.** A IEEE Access publica foto e mini-bio dos autores.
   Mande uma foto tipo documento (rosto, fundo neutro, boa resolução) que eu
   coloco no lugar de "PLACE PHOTO HERE". Na submissão inicial pode ir sem.
6. **Bolsa/financiamento.** Se você tem bolsa (CAPES, FAPERJ, CNPq...), me
   avise o número do processo: é obrigatório declarar no artigo.
7. **Coautoria.** Hoje o artigo tem autor único. Se a Milena (ou seu
   orientador) for entrar como coautor(a), precisa ser **antes** da
   submissão, com contribuição real (revisão crítica, discussão dos
   resultados). O resumo em português está em `submissao/resumo_pt.md`.

## Passo a passo no ScholarOne

1. Acesse o sistema de submissão da IEEE Access (link em
   ieeeaccess.ieee.org → "Submit Manuscript") e crie a conta com o e-mail
   institucional.
2. Tipo de artigo: **Research Article**.
3. Título, resumo e palavras-chave: copiar de `paper/sections/abstract.tex`
   (o resumo tem menos de 250 palavras, como a revista exige).
4. Área/tópicos: Power and Energy; Machine Learning / Artificial Intelligence.
5. Arquivos: `paper/main.pdf` como manuscrito principal. Se pedirem os
   fontes, um zip da pasta `paper/` (main.tex, sections/, tables/,
   figures/, references.bib, ieeeaccess.cls e logos).
6. Carta de apresentação: `submissao/cover_letter.txt`.
7. Declarações: artigo original, não submetido em outro lugar, sem conflito
   de interesse. O uso de IA já está declarado nos Agradecimentos, dizendo em
   que partes foi usada, como pede a IEEE.
8. Revisores sugeridos (se pedirem): posso montar uma lista com autores dos
   artigos citados (Lago, Weron, Ziel, Hong, Gibbs). Basta pedir.

## Depois da aceitação

- Licença: escolha **CC BY 4.0** (open access).
- APC: no formulário de pagamento (RightsLink), a afiliação CEFET/RJ e o
  e-mail institucional devem fazer aparecer a cobertura do acordo CAPES–IEEE
  (válido para artigos aceitos de 05/11/2024 a 30/10/2027). **Não pague
  nada**: se a isenção não aparecer, me avise antes.

## Comentários dos revisores

Se vier revisão, me mande o PDF ou o texto dos comentários. Eu preparo a
versão revisada e a carta de resposta ponto a ponto.

## Riscos conhecidos (e como o artigo os trata)

| Possível crítica | Resposta no artigo |
|---|---|
| "Os modelos não usam clima" | Limitação declarada. Torna a comparação conservadora. O ONS usa clima e mesmo assim perde em 3 de 4 subsistemas. |
| "Não se sabe o horário de corte do ONS" | Teste de robustez sem nenhum dado de D-1: ainda vence no N e NE. |
| "Só Brasil" | Enquadramento geral (crescimento de solar atrás do medidor). A replicação em outros operadores fica como trabalho futuro. |
| "O ONS corrigiu o viés em 2026" | Mostrado e analisado: a vantagem continua em N, NE e S depois de jul/2026. |
| "A referência X não existe" | As 43 referências foram conferidas no Crossref ou na página oficial da conferência ou do órgão. |
