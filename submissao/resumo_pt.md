# Resumo em português (para revisão)

**Previsão de carga do dia seguinte sob crescimento acelerado da geração solar
atrás do medidor: um benchmark aberto contra a previsão oficial de um operador
de sistema no Brasil**

## Problema

Os painéis solares nos telhados (MMGD) mudam o formato da curva de carga que o
operador enxerga. No SIN, essa geração passou de cerca de 5% da carga do
meio-dia em 2021 para cerca de um terço em 2026. Modelos de previsão abertos
quase nunca são comparados com a previsão que o operador de fato usa.

## O que foi feito

- Dados 100% públicos do ONS: curva de carga horária (2012–2026), carga global
  verificada com estimativa de MMGD (2020–2026) e carga programada, que é a
  previsão oficial do dia seguinte (2021–2026).
- Previsão emitida ao meio-dia de D-1, usando dados até as 10h de D-1
  (horizontes de 14 a 37 h). É o mesmo cenário de informação do operador.
- Modelos: naïve sazonal, ARX (24 regressões lineares, uma por hora, com ridge),
  LightGBM, rede neural (MLP) e a média dos três (ENS).
- Teste: jan/2024 a nov/2026 (1.039 dias), nunca usado para ajustar os
  modelos. Testes de Diebold–Mariano com correção de Holm.

## Resultados principais

1. **Contra o ONS:** o ENS, sem nenhum dado de clima, teve MAPE de 2,44% contra
   2,90% da previsão oficial (16% menor). A vantagem é estatisticamente
   significativa no N (1,96 vs 2,62), NE (1,96 vs 2,71) e S (3,57 vs 3,92). No
   SE/CO os dois empatam (2,26 vs 2,33).
2. **Por quê:** a previsão oficial subestima sistematicamente a carga entre 10h
   e 14h. Esse viés piorou de 2024 para 2026 junto com o crescimento da MMGD
   (por exemplo, de −0,8% para −3,0% no NE). O viés dos modelos abertos ficou
   perto de zero.
3. A partir de jul/2026 o viés do ONS diminuiu (provável revisão de
   procedimento), mas o ENS continuou melhor no N, NE e S.
4. Mesmo sem nenhum dado de D-1, o ARX ainda vence o ONS no N e no NE.
5. **Janela e retreino:** o tamanho da janela de treino importa mais do que a
   frequência de retreino. Retreinar todo dia ou todo trimestre dá quase o
   mesmo resultado quando a carga é normalizada pelo nível recente.
6. **Normalização:** dividir a carga pelo nível da última semana é essencial
   para o LightGBM (o MAPE sobe de 2,96% para 3,23% sem ela).
7. **Intervalos de previsão:** o LightGBM quantílico é muito otimista (o
   intervalo de 90% cobre só 69%). Os quantis empíricos dos erros recentes com
   inferência conformal adaptativa (ACI) atingem a cobertura nominal (80,3% e
   90,1%).
8. **Feriados e dias especiais** têm erro de 1,2 a 2 vezes o de dias úteis
   comuns. Pontes e fim de ano são bem tratados pelo calendário.

## Limitações declaradas

Os modelos não usam clima. O horário exato de corte da previsão do ONS não é
publicado. A carga global verificada contém uma estimativa de MMGD feita pelo
próprio ONS. Feriados regionais não são modelados. O estudo cobre um único
sistema.

## Pontos para a revisão

- O enquadramento e a explicação do viés (seção VI-A) fazem sentido para quem
  conhece a programação diária do ONS?
- Alguma referência brasileira importante sobre previsão de carga do ONS que
  deveria ser citada?
- Alguma afirmação sobre o processo do ONS que esteja imprecisa?
