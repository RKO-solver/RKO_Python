# Linguagem nativa de decoders RKO — versão 1

Status: contrato da V1, agosto de 2026.

## Propósito

A linguagem nativa V1 é um subconjunto explícito de Python para escrever o
trecho quente de um ambiente RKO. Um ambiente pertencente a esse subconjunto
pode ser transformado em `Problem.h` e avaliado pelo RKO C++ sem chamar o
interpretador Python durante a busca.

Ela não pretende aceitar Python arbitrário. O contrato é deliberadamente
pequeno, determinístico e adaptável: quando uma construção está fora do perfil,
o compilador deve rejeitá-la com a localização e cadeia disponíveis, motivo e,
quando aplicável, uma sugestão objetiva de adaptação. Nunca há fallback
silencioso para Python nem substituição por uma operação apenas "parecida".

O programa nativo observado pelo RKO é:

```python
solution = env.decoder(keys)
objective = env.cost(solution, False)
```

O C++ gerado expõe esse programa como uma única avaliação:

```cpp
double evaluate(keys, frozen_instance);
```

Ao final da otimização, as melhores chaves retornam ao Python. O ambiente
original pode então reconstruir a solução e produzir arquivos, gráficos e
relatórios fora do trecho nativo.

## Contrato do ambiente

Um ambiente V1 deve atender aos seguintes requisitos:

- ser uma instância Python já inicializada pelo reader e por `__init__`;
- possuir `tam_solution: int`;
- definir `decoder(self, keys: list[float])`;
- definir `cost(self, solution, final_solution: bool = False) -> float`;
- ter anotações de parâmetros e retorno em `decoder`, `cost` e helpers
  alcançáveis;
- produzir o mesmo resultado para as mesmas chaves e o mesmo snapshot;
- tratar os atributos de `self` como somente leitura durante a avaliação;
- manter todo estado temporário em variáveis e listas locais.

O tipo concreto retornado por `decoder` deve ser aceito pelo parâmetro
`solution` de `cost`. A solução é interna ao código gerado e não precisa de uma
ABI pública. Na V1, os retornos usuais são `list[int]` e `list[float]`.
As assinaturas são verificadas exatamente: `decoder` recebe um único
`list[float]`; `cost` recebe o mesmo tipo retornado pelo decoder, seguido de
`bool` com default literal `False`, e retorna `float`.

O adaptador C++ chama `cost(decoded, False)`, mas o frontend V1 atual analisa o
método `cost` inteiro. Não há promessa de especialização, propagação dessa
constante ou eliminação de `if final_solution:`. Portanto, I/O, gráficos e
relatórios devem estar em outro método Python, fora de `cost`; colocá-los apenas
em um ramo `if final_solution:` ainda torna o ambiente incompatível com a V1.

## Tipos

### Tipos escalares

| Python | Representação nativa | Regra V1 |
|---|---|---|
| `bool` | `bool` | `True` ou `False` |
| `int` | inteiro assinado de 64 bits | valores e resultados devem caber em `int64` |
| `float` | IEEE-754 binário de 64 bits | mesma precisão de um `float` CPython |
| `None` | `void` | somente retorno sem valor de helper |

Conversões explícitas `int(x)`, `float(x)` e `bool(x)` só são aceitas entre os
tipos escalares acima. A V1 não implementa inteiros Python de precisão
arbitrária. Dados congelados fora da faixa de `int64` são rejeitados. Literais,
conversões e resultados intermediários dentro dessa faixa, sem overflow, são
uma precondição do programa; o frontend atual não realiza uma prova geral de
overflow.

Armazenamento, retorno e passagem de parâmetros usam o tipo concreto exato:
uma expressão `int` não é implicitamente armazenada como `float`. Use
`float(value)` quando essa conversão for intencional. Operações aritméticas
mistas ainda produzem `float`, como no Python. `min`, `max` e comparações
exigem operandos do mesmo tipo concreto para evitar perda implícita de inteiros
acima de `2**53`.

### Containers estruturais

`list[T]` é aceita quando todos os elementos têm o mesmo tipo `T`. O tipo pode
ser aninhado, por exemplo:

```python
list[float]
list[int]
list[list[float]]
```

Listas aninhadas podem ser retangulares ou irregulares, desde que cada elemento
possua o mesmo tipo estrutural. Listas vazias locais precisam de anotação:

```python
order: list[int] = []
```

Um atributo congelado vazio não oferece evidência suficiente para inferência
na V1 e deve possuir esquema explícito futuramente; por enquanto ele é
rejeitado.

`tuple[T0, T1, ...]` representa uma tupla de comprimento e tipos fixos. Tuplas
podem ser heterogêneas, podem aparecer em atributos congelados, anotações,
literais e retornos, e podem ser indexadas apenas por uma constante inteira
válida. Um literal de tupla que contenha lista é rejeitado, pois poderia ocultar
um alias mutável. Na V1 atual, tuplas não são iteráveis em `for`/`enumerate` e
não são mutáveis.

Arrays e escalares NumPy usados como dados congelados são normalizados por
`.tolist()`/`.item()`. O que define sua aceitação é o valor normalizado: ele
precisa resultar exclusivamente em escalares V1, tuplas ou listas homogêneas
não vazias em cada nível. Assim, ranks maiores que dois e dtypes numéricos
menores também podem ser capturados, desde que todos os valores normalizados
sejam finitos e inteiros caibam em `int64`. A V1 não preserva layout, strides,
views, dtype exato nem aliases NumPy; preserva somente os valores do snapshot.

As chaves recebidas do RKO são vistas sempre como uma sequência contígua de
`float64`, de comprimento `tam_solution`. No perfil normal do RKO, cada chave é
finita e pertence a `[0, 1)`. `NaN` e infinitos nas chaves estão fora do domínio
V1.

Todos os valores `float` que chegam pelo snapshot e todos os resultados
intermediários do programa V1 devem ser finitos. NaN e infinitos ficam fora do
domínio desta versão. Da mesma forma, operações inteiras têm como precondição
não produzir overflow de `int64`.

### Tipos não aceitos na V1

Não fazem parte do trecho nativo:

- `str`, `bytes` e identificadores textuais;
- `dict`, `set` e seus derivados;
- classes Python arbitrárias e dataclasses;
- `Optional`, unions e valores `None` usados em cálculos ou containers;
- listas heterogêneas como `["SKU-1", 10.0, 4.0]`;
- objetos NumPy que não sejam normalizáveis para os tipos V1.

Esses valores podem continuar existindo no ambiente, desde que não sejam
alcançados por `decoder`, `cost` ou seus helpers. Identificadores como SKU devem
ser normalizados pelo reader para IDs inteiros e, se necessário, convertidos de
volta no relatório Python.

## Dados da instância

O reader e `__init__` executam em Python. O compilador captura somente os
atributos `self.<nome>` lidos pelo programa alcançável e produz:

```text
InstanceSchema   nomes, tipos e formas esperadas
InstanceSnapshot valores concretos da instância
```

Na integração V1, o snapshot textual é escrito em `instance.rko-data`, com o
magic `RKO_NATIVE_DATA_V1`, e carregado uma vez por `ReadData`. O código gerado
fica em `Problem.h`. O compilador também produz o fonte e AST capturados,
`typed_ir.json`, `instance_schema.json`, `compile_report.json` e
`manifest.json`. Alterar somente os valores de uma instância compatível com o
mesmo esquema não exige mudar o fonte do decoder, embora uma chamada de
compilação ainda possa regenerar os artefatos.

Regras para atributos capturados:

- escalares, listas e tuplas são copiados por valor para o snapshot;
- todos os dados capturados são profundamente somente leitura no C++;
- o valor capturado deve ser acíclico; identidade e aliases entre containers
  não são preservados pelo codec, que serializa seus valores;
- atributos não alcançados são ignorados, inclusive nomes, caches e dados de
  visualização;
- mutar `self`, um atributo de `self` ou uma lista obtida diretamente dele é
  proibido no hot path;
- para consumir ou modificar dados, o programa deve construir uma lista local.

Exemplo válido:

```python
remaining = self.quantities.copy()
remaining[index] -= amount
```

Exemplo inválido:

```python
self.quantities[index] -= amount
```

## Funções alcançáveis

Além das duas raízes, a V1 compila recursivamente métodos auxiliares chamados
estaticamente por `self`:

```python
def _priority(self, job: int, key: float) -> float:
    return key + self.bias[job]

def decoder(self, keys: list[float]) -> list[int]:
    # self._priority é resolvido e compilado como helper nativo.
    ...
```

Regras:

- o método deve ser um `def` definido diretamente no corpo da mesma classe que
  contém `decoder` e `cost`; métodos herdados ainda não são descobertos pela V1;
- métodos alcançáveis não podem ter decorators, parâmetros positional-only,
  keyword-only, `*args` ou `**kwargs`;
- argumentos e retorno devem usar tipos V1;
- um helper pode modificar um parâmetro `list[T]` quando o chamador fornece uma
  lista local mutável pertencente à avaliação atual; o frontend marca e propaga
  parâmetros mutados, mas esse contrato de ownership continua sendo uma
  obrigação do programa V1;
- parâmetros ligados a `keys`, `solution` e dados alcançados diretamente de
  `self` são somente leitura; para modificá-los é necessário criar uma cópia
  local explícita;
- dispatch dinâmico, monkey patching, `getattr` e escolha de callable em runtime
  não são aceitos;
- recursão, closures, funções internas e `nonlocal` ficam fora da V1;
- funções livres do mesmo módulo não são aceitas na V1; devem ser
  transformadas em helpers de `self` ou em operação de provider reconhecida;
- chamadas de helpers usam somente argumentos posicionais e devem fornecer
  todos os parâmetros explicitamente; defaults e keywords de helpers não são
  aplicados pelo frontend V1.

## Statements aceitos

A V1 aceita os seguintes statements dentro das funções alcançáveis:

```python
# declaração/atribuição local
x = expression
x: float  # declara o tipo, mas não inicializa o nome
x: float = expression
x += expression
x -= expression
x *= expression
x /= expression
x //= expression
x %= expression

# mutação de lista local
values.append(expression)
item = values.pop()
item = values.pop(index)
values[index] = expression
values[index] += expression

# controle de fluxo
if condition:
    ...
else:
    ...

for index in range(stop):
    ...

for index in range(start, stop):
    ...

for index in range(start, stop, step):
    ...

for index, value in enumerate(values):
    ...

for value in values:
    ...

while condition:
    ...

break
continue

return expression
return
```

Atribuições têm um único alvo: nome local ou subscript de lista. Desempacotamento
em atribuições e atribuições encadeadas não fazem parte da V1; o
desempacotamento de dois nomes usado por `enumerate` é aceito apenas no alvo do
`for`. Condições de `if`/`while` e operandos de `and`/`or` precisam ser `bool`;
truthiness implícita de números e containers não é usada.

`range` exige de um a três argumentos inteiros. O `step`, quando presente, deve
ser um literal inteiro constante diferente de zero. Seus argumentos são
avaliados uma única vez antes do loop, como no Python.
`break` e `continue` só podem aparecer dentro do loop léxico correspondente.
Um `while` deve depender apenas de estado V1 e sua terminação é uma
precondição do ambiente. Docstrings e `pass` são ignorados. A V1 não aceita
`for ... else`, `while ... else`, `try`, `raise`, `assert`, `with`, `yield`,
`async`, imports locais ou definição de funções/classes dentro do hot path.

Os valores produzidos por `for value in lista` e por `enumerate(lista)` devem
ser escalares de leitura; iterar diretamente sobre elementos lista/tupla é
rejeitado. O iterável não pode ser modificado durante o loop. Para alterar uma
linha aninhada, use um índice explícito sobre uma lista local. O alvo do `for`
deve ser um nome novo e não fica disponível depois do loop.

Toda variável precisa ter sido inicializada em todos os caminhos antes de ser
lida. O frontend intersecta o estado de inicialização dos dois ramos de `if` e,
conservadoramente, não considera inicializadas depois de um loop as variáveis
criadas somente em seu corpo. Funções com retorno não-`None` devem terminar com
`return` em todo caminho estrutural verificável.

## Expressões aceitas

### Valores e operadores

- literais `bool`, `int` e `float`;
- nomes locais e leituras `self.<atributo>`;
- indexação de listas com inteiros e de tuplas por constante inteira;
- listas escalares, tuplas sem listas e list comprehensions simples;
- `+`, `-`, `*`, `/`, `//` e `%` em escalares compatíveis;
- repetição de lista pelo formato `[valor] * quantidade`, como `[0.0] * n`;
- `+x`, `-x` e `not x`;
- `<`, `<=`, `>`, `>=`, `==` e `!=`;
- `and` e `or`, preservando short-circuit;
- expressão condicional `a if condition else b`.

Índices negativos de listas são aceitos e normalizados como no Python: `-1`
denota o último elemento. Um índice que permaneça fora de `[-len(lista),
len(lista)-1]` lança erro no executável nativo. Slices são rejeitados. Índices
de tupla precisam ser constantes válidas. Divisão por zero e acesso fora dos
limites estão fora do domínio de uma execução válida; acessos nativos a listas
também verificam os limites e lançam erro.

List comprehensions possuem exatamente um gerador, sem filtro:

```python
selected = [1 if key > 0.5 else 0 for key in keys]
copied = [value for value in self.quantities]
```

Comprehensions têm exatamente um gerador síncrono, sem filtro, com alvo que seja
um único nome. O iterável precisa ser uma expressão já tipada como `list[T]`;
`range(...)`, `enumerate(...)`, tuplas e outros iteráveis não são aceitos dentro
de comprehension na V1 atual. Comprehensions aninhadas, assíncronas ou contendo
`if` após o gerador também são rejeitadas.

### Builtins

As chamadas V1 são:

```text
len(list|tuple) -> int
int(scalar) -> int
float(scalar) -> float
bool(scalar) -> bool
abs(int|float) -> mesmo tipo
min(escalar, escalar) -> tipo numérico comum
max(escalar, escalar) -> tipo numérico comum
round(float) -> int
round(float, dígitos_constantes_não_negativos) -> float
range(...)
enumerate(list)
```

`min` e `max` recebem exatamente dois argumentos. Na segunda forma de `round`,
`ndigits` deve ser um literal inteiro constante e não negativo. O backend usa
conversão decimal C++20 com `to_chars`/`from_chars`, evitando a dupla perda de
precisão de `round(x * 10**n) / 10**n`. Keywords não são aceitas nessas
chamadas.

O provider `math` atual aceita exatamente uma expressão numérica posicional em:

```text
math.sqrt  math.exp   math.log
math.sin   math.cos   math.tan
math.floor math.ceil
```

Aliases provenientes de imports de módulo, como `from math import sqrt`, são
resolvidos pelo mapa de imports. `math.sqrt`, `exp`, `log`, `sin`, `cos` e
`tan` produzem `float`; `math.floor` e `math.ceil` produzem `int`. Domínio
inválido, resultado não finito e conversão fora de `int64` causam erro explícito
no runtime nativo.

`sorted`, `sum`, `any`, `all`, `zip`, lambdas e argumentos `key=` não fazem
parte da V1. Seus equivalentes podem ser escritos com loops ou adicionados
posteriormente como operações semânticas testadas.

### Listas locais

São aceitas:

```text
list.append(valor)
list.copy()
list.pop()
list.pop(índice)
list[index]
list[index] = valor
len(list)
[valor] * quantidade_inteira
```

`copy()` cria uma cópia rasa e uma lista local com ownership explícito. A V1
rejeita atribuições como `local = outra_lista`, que manteriam alias no Python e
copiariam valor em C++. A V1 atual aceita `copy()` somente quando o elemento não
é outra lista; `nested.copy()` é rejeitado para não confundir a cópia rasa do
Python com uma cópia profunda de `std::vector`. A repetição implementada é o
formato literal de um elemento `[valor] * quantidade`; quantidade negativa
produz uma lista vazia, como no Python. O valor repetido não deve conter uma
lista mutável, pois isso criaria alias observável no Python.

`append` e `pop` são permitidos somente sobre listas locais mutáveis da
avaliação. `pop()` equivale a `pop(-1)`; `pop(índice)` aceita índices negativos
com a mesma normalização da indexação. Usá-los diretamente em `self.campo`, em
`keys`, em `solution` ou em outro parâmetro somente leitura está fora da V1.
Helpers podem receber e modificar uma lista local por referência, preservando a
mutação para o chamador.

Chamadas com efeito (`pop` e helper que muta parâmetro) não podem ficar
aninhadas em aritmética, comparações ou argumentos de outra chamada. Elas devem
ser uma instrução completa, o lado direito completo de uma atribuição ou um
`return`. Isso torna a ordem Python explícita antes de gerar expressões C++:

```python
item = values.pop()
score = item + bonus
```

`insert`, `remove`, `clear`, `sort`, concatenação, slices e cópias implícitas não
fazem parte da V1. Isso mantém explícitos tamanho, alias e custo das mutações.

### NumPy

A única operação NumPy normativa na V1 é:

```python
np.argsort(values, kind="stable").tolist()
```

O `.tolist()` final é opcional na fonte V1: o provider já representa o retorno
nativo como `list[int]`.

Contrato:

- `values` é unidimensional e contém `float64` finitos;
- `axis` é o padrão `-1`;
- `kind="stable"` é literal e obrigatório;
- `order` não é usado;
- o retorno é `list[int]`;
- índices empatados preservam a ordem original.

O resolvedor atual usa os imports no arquivo-fonte da classe. Formas diretas
como `import numpy as np` e `from numpy import argsort` podem ser convertidas à
identidade canônica `numpy.argsort`; reexports, shadowing e bindings dinâmicos
não são garantidos. `np.argsort(values)` sem `kind="stable"` é rejeitado porque
não possui o mesmo contrato de estabilidade. `axis`, `order` ou qualquer outro
keyword adicional também é rejeitado. Outras operações NumPy ficam fora da V1
até receberem provider e testes diferenciais próprios.

## Efeitos e concorrência

Uma função V1 pode:

- calcular valores puros;
- ler dados congelados de `self`;
- criar e modificar escalares e listas locais;
- passar listas locais para helpers que as modificam de forma analisável.

Ela não pode:

- modificar `self` ou dados congelados;
- depender de estado persistente entre avaliações;
- realizar I/O, logging, gráficos ou serialização;
- ler relógio, aleatoriedade ou variáveis de ambiente;
- usar reflexão ou importar módulos dinamicamente;
- capturar ou modificar globais;
- chamar Python ou adquirir o GIL durante a busca nativa.

`evaluate` deve ser reentrante. Cada chamada possui seus próprios temporários;
o snapshot é compartilhado somente para leitura. Dessa forma, duas avaliações
podem ocorrer simultaneamente nas threads do RKO C++.

## Semântica e validação

O compilador deve preservar a ordem de avaliação e a semântica Python declarada
por esta especificação. Em particular:

- `/` produz ponto flutuante;
- `//` e `%` preservam as regras de piso e sinal do Python;
- índices negativos de listas e de `pop` são normalizados como no Python;
- `and` e `or` usam short-circuit;
- `np.argsort(..., kind="stable")` preserva empates;
- conversões e `round` seguem o contrato do provider selecionado.

Toda classe aceita deve passar por comparação diferencial:

```text
Python: env.cost(env.decoder(keys), False)
    ==
C++:    evaluate(keys, snapshot)
```

A comparação deve incluir chaves fixas, aleatórias e com empates. A API
`compile_environment` gera os artefatos e o relatório, mas não executa essa
comparação automaticamente. A API de ponta a ponta `optimize_environment`
compila, executa o RKO, exige por padrão chaves exatas, reconstrói a solução no
Python e reavalia as chaves no evaluator C++. A suíte de validação continua
necessária para cobrir o domínio do decoder antes do uso em produção.

O RKO C++ empacotado emite, além do relatório humano legado, uma linha
`RKO_RESULT_V1` com `max_digits10`. Versões externas sem esse protocolo caem em
parser legado marcado como lossy; a API de alto nível recusa esse fallback por
padrão porque três casas de chave podem alterar thresholds e permutações.

## Diagnósticos

Uma falha de compatibilidade é normal. `analyze_environment` a devolve no
relatório, e `compile_environment` grava o relatório e lança
`NativeCompileError`. O diagnóstico serializado atual contém:

```text
code          código estável da família RKO-NATIVE-*
message       descrição da regra violada
span          filename, line, column e limites, quando disponíveis
call_chain    função em análise/cadeia disponível
adaptation    mudança sugerida, quando disponível
```

Famílias iniciais:

| Código | Significado |
|---|---|
| `RKO-NATIVE-SOURCE-*` | fonte ou raízes do ambiente indisponíveis |
| `RKO-NATIVE-SYNTAX-*` | statement ou expressão fora da gramática |
| `RKO-NATIVE-TYPE-*` | anotação/tipo ausente, heterogêneo ou incompatível |
| `RKO-NATIVE-NAME-*` | nome local não resolvido |
| `RKO-NATIVE-CALL-*` | alvo ou assinatura da chamada não suportado |
| `RKO-NATIVE-EFFECT-*` | mutação ou alias proibido |
| `RKO-NATIVE-CONTROL-*` | fluxo sem retorno total comprovável |
| `RKO-NATIVE-DATA-*` | atributo não serializável ou snapshot inválido |
| `RKO-NATIVE-VALUE-*` | literal fora do domínio V1 |
| `RKO-NATIVE-NUMPY-*` | assinatura NumPy fora do provider V1 |
| `RKO-NATIVE-CPP-*` | tipo verificado sem representação no backend |

Exemplo:

```text
code: RKO-NATIVE-NUMPY-003
message: V1 argsort requires the explicit keyword kind='stable'
span: {filename: environment.py, line: 42, column: 16, ...}
call_chain: [decoder]
adaptation: use np.argsort(values, kind="stable").tolist()
```

O frontend V1 atual é fail-fast: o relatório de falha contém o primeiro bloqueio
encontrado. Se não houver erros, o relatório lista os helpers compilados, os
atributos congelados, seus tipos e os providers detectados.

## Fora de escopo e evolução

Uma construção não suportada não é considerada impossível. Ela apenas não
pertence ao contrato V1. A evolução ocorre adicionando uma regra completa de
resolução, tipo, IR, backend, diagnóstico e testes diferenciais — nunca por uma
condição especial para um ambiente.

Exemplos de extensões posteriores naturais:

- records nomeados para soluções estruturadas;
- `dict[int, T]` e strings congeladas;
- outras mutações estruturais além de `append` e `pop`;
- providers adicionais de `math` e NumPy;
- funções livres e closures convertidas;
- snapshot binário e cache de build por schema.

KP, TSP, scheduling e Ball são corpus de validação. Nenhum nome de classe,
arquivo ou problema pode aparecer como caso especial no compilador.
