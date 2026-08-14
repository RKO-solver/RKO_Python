# Compilador nativo de ambientes RKO

Status: proposta de arquitetura, agosto de 2026.

## Decisão central

O projeto não deve tentar compilar Python inteiro e não deve conter regras por
nome de ambiente. Ele deve compilar o programa alcançável a partir de duas
raízes de um ambiente já inicializado:

```python
solution = env.decoder(keys)
objective = env.cost(solution, False)
```

O alvo nativo principal é a função fundida:

```text
evaluate(keys, frozen_instance) -> float64
```

A solução decodificada permanece um valor interno tipado. Ao terminar a busca,
o RKO C++ devolve as melhores chaves em precisão integral, e o Python original
executa novamente `decoder` e o caminho de relatório final. Assim não é
necessário inventar uma ABI universal para soluções heterogêneas.

O compilador é de programa inteiro e especializado pelo tipo concreto do
ambiente, pelo grafo de chamadas alcançável e pelo esquema da instância. Ele
deve sempre produzir um relatório de capacidades, inclusive quando a
compilação falhar.

## Objetivos

- Cobrir uma gama crescente de decoders e custos reais sem casos especiais por
  problema.
- Compilar recursivamente métodos do ambiente, funções auxiliares, closures e
  funções Python puras alcançáveis.
- Permitir providers versionados para builtins, `math`, NumPy e bibliotecas
  nativas.
- Executar readers e inicialização em Python e transportar somente um snapshot
  tipado dos dados alcançáveis.
- Preservar semântica ou rejeitar explicitamente. Nunca substituir uma operação
  por outra "parecida" de forma silenciosa.
- Gerar código determinístico, reentrante e seguro para chamadas concorrentes
  do RKO C++.
- Explicar exatamente por que um ambiente não compila e como ele pode ser
  adaptado sem alterar o algoritmo.

## Não objetivos

- Compatibilidade universal com Python.
- Compilar `__init__`, readers, gráficos, persistência e relatórios por padrão.
- Aceitar reflexão arbitrária, `eval`, `exec`, monkey patching ou dispatch que
  não possa ser resolvido estaticamente.
- Usar callbacks Python ocultos no modo nativo estrito.
- Inferir equivalência somente porque dois custos coincidiram em uma amostra.

## Fatos do contrato atual

### Python

- `keys` nasce normalmente como `numpy.ndarray[float64]`, mas também circula
  como `list`; a fronteira do compilador deve normalizá-la para uma sequência
  contígua de `float64` com tamanho `tam_solution`.
- A solução produzida pelo decoder é opaca para o RKO.
- O custo efetivo pode ser `int`, `float` ou escalar NumPy, mas precisa ser
  convertido e validado como `float64` para o backend.
- O hot path usa `cost(solution, False)`. I/O e visualização associados a uma
  solução final devem continuar no Python.
- Anotações existentes são dicas, não fonte exclusiva de verdade.

### C++

O RKO original exige uma interface de fonte em `Problem.h`:

```cpp
struct TProblemData {
    int n;
    // restante livre
};

void ReadData(char* path, TProblemData& data);
double Decoder(TSol& solution, const TProblemData& data);
void FreeMemoryProblem(TProblemData& data);
```

`ReadData` roda uma vez. `Decoder` é uma função quente e pode ser chamada
simultaneamente por várias threads. `TProblemData` é compartilhado por
referência constante durante toda a busca.

## Fluxo de compilação

```text
env Python já inicializado
        |
        v
descoberta de fonte, bindings e grafo de chamadas
        |
        v
especialização conservadora + closure conversion
        |
        v
inferência de tipos + análise de efeitos
        |
        +--------------------+
        |                    |
        v                    v
RKO-IR verificada      InstanceSchema
        |                    |
        v                    v
interpretador IR       InstanceSnapshot
        |                    |
        +---------+----------+
                  v
           providers + backend
                  |
                  v
       módulo C++ + adaptador Problem.h
                  |
                  v
          build content-addressed
                  |
                  v
       validação diferencial no WSL
```

## 1. Descoberta e resolução do programa

As raízes são `type(env).decoder` e `type(env).cost`. O resolvedor deve seguir
transitivamente:

- métodos ligados de `self`, respeitando MRO;
- funções de módulo e imports relativos;
- funções capturadas por closures;
- argumentos default;
- globals imutáveis alcançados;
- builtins e funções de bibliotecas.

A identidade deve ser resolvida a partir do binding Python real, e não somente
do texto como `np.argsort`. Imports renomeados, reexports e shadowing precisam
ser tratados corretamente.

Alvos dinâmicos podem ser desvirtualizados quando o objeto concreto e o binding
são estáveis no momento da compilação. Caso contrário, o relatório deve mostrar
o call site e os alvos que não puderam ser determinados.

Cada nó do grafo carrega:

- objeto resolvido e origem;
- arquivo e source span;
- fingerprint do fonte;
- closure e globals capturados;
- assinaturas especializadas;
- efeitos;
- dependências de providers.

Recursão é representada por componentes fortemente conectados do grafo; não
deve ser expandida infinitamente.

## 2. Especialização conservadora

O ambiente concreto fornece informações úteis, mas valores runtime não podem
ser usados como constantes indiscriminadamente.

Classificações necessárias:

```text
CompileTimeConstant   valor seguro para partial evaluation
FrozenInput           dado da instância, somente leitura
EvaluationLocal       estado novo para cada avaliação
ForbiddenSharedState mutação persistente ou ambígua
```

Casos como `final_solution=False` são constantes do entrypoint nativo e podem
eliminar ramos de gráficos, JSON e arquivos antes da análise de compatibilidade.

Uma escrita em `self.attr` só pode ser privatizada quando a análise provar que
o atributo representa temporário daquela avaliação. Mutação deliberadamente
persistente entre avaliações deve ser rejeitada, pois altera a função objetivo
e cria corrida no RKO paralelo.

## 3. Sistema de tipos

A IR precisa de tipos imutáveis e estruturais, não dicionários ad hoc.

Tipos iniciais:

```text
Bool
Int                    semântica Python, representação escolhida pelo backend
Float64
String
NoneType
Optional[T]
Tuple[T0, T1, ...]
Record[field: T, ...]
List[T]
Dict[K, V]
Set[T]
Array[dtype, rank, layout]
Slice
Range
Function[args -> result]
```

Fontes de restrições:

- anotações;
- valores concretos do snapshot;
- operações observadas na AST;
- assinaturas dos providers;
- fluxo de controle;
- tipos de retorno inferidos transitivamente.

O compilador deve possuir escopos lexicais, análise de atribuição definida e
merge de tipos nos joins do fluxo. Funções são monomorfizadas por assinatura
concreta quando necessário.

Containers heterogêneos de tamanho e posições estáveis podem virar `Tuple` ou
`Record`. Uma lista heterogênea que sofre mutação estrutural incompatível deve
ser diagnosticada; não se deve inventar um cast.

## 4. RKO-IR semântica

A IR deve representar semântica Python, não operadores textuais C++.

Exemplos:

```text
PyTrueDiv
PyFloorDiv
PyModulo
PyRound
SequenceGetItem
SequenceSlice
DictGetItem
ListAppend
ListPop
NumPyArgsort
```

Isso evita divergências silenciosas como:

- `int / int`;
- módulo com operandos negativos;
- índices negativos;
- regras de `round`;
- truthiness e short-circuit;
- ordem de dicionários;
- overflow de inteiros;
- ordenação de `NaN` e empates.

A IR deve manter controle de fluxo estruturado inicialmente (`if`, `for`,
`while`, `break`, `continue`, regions e closures convertidas). SSA pode ser
introduzida internamente para análise e otimização, mas não é requisito para o
primeiro backend.

Toda IR passa por um verificador obrigatório antes de ser interpretada ou
emitida.

## 5. Análise de efeitos

Cada função e operação declara efeitos:

```text
Pure
ReadFrozen
MutateLocal
MutateShared
IO
Random
Clock
Reflection
MayRaise
```

O modo nativo estrito aceita `Pure`, `ReadFrozen` e `MutateLocal`. Outros
efeitos precisam ser eliminados por especialização, tratados por uma regra
explícita ou rejeitados.

Providers também declaram determinismo, thread-safety, uso de estado global,
inicialização e possibilidade de paralelismo interno. Paralelismo interno deve
ficar desativado inicialmente para evitar oversubscription com o OpenMP do RKO.

## 6. Interpretador de referência

Antes do backend C++, a nova IR precisa de um interpretador em Python. Para
cada ambiente e conjunto de chaves:

```text
env.cost(env.decoder(keys), False)
              ==
RkoIrInterpreter.evaluate(keys, snapshot)
```

Uma feature só avança ao backend depois dessa equivalência. Isso separa erros
do frontend/tipagem de erros do gerador C++.

## 7. Dados de entrada

O reader e o `__init__` continuam em Python. O compilador captura apenas os
atributos alcançáveis pelo programa nativo.

Devem existir dois artefatos versionados:

```text
InstanceSchema
  nomes, tipos, ranks, layouts, mutabilidade e hash

InstanceSnapshot
  valores concretos validados contra o schema
```

O formato precisa ter:

- versão do codec;
- hash do schema;
- comprimentos e encoding explícitos para strings;
- validação de limites e shapes;
- detecção de ciclos e aliases relevantes;
- mensagem de erro por campo.

O modo padrão separa código e dados: um decoder/esquema é compilado uma vez e
novas instâncias geram apenas snapshots. Embutir valores em C++ é uma
otimização opcional para instâncias pequenas e especializadas, não a arquitetura
base.

Dados congelados ficam profundamente imutáveis. Cópias consumidas por `pop`,
filas, resultados e outros temporários pertencem ao `EvalState` criado dentro
de cada chamada.

## 8. Providers de funções e bibliotecas

Cada chamada resolvida segue uma destas rotas:

```text
função Python pura       -> compilação recursiva
builtin/math             -> provider semântico
fonte C/C++ disponível   -> compilação/link provider
API nativa estável       -> link provider
compilador externo       -> backend/provider versionado
sem rota válida          -> unsupported
```

Um provider é selecionado por identidade canônica e assinatura, não apenas por
nome textual. Seu manifesto contém:

```text
distribuição e versões suportadas
callable resolvido
tipos, rank, dtype e argumentos constantes exigidos
semântica declarada
efeitos, determinismo e thread-safety
headers, fontes, bibliotecas e flags
fingerprints dos artefatos
suíte de conformidade
```

O relatório registra qual provider implementou cada operação. Não há fallback
silencioso entre algoritmos diferentes.

### NumPy

O provider NumPy pode ter mais de uma estratégia, escolhida explicitamente:

1. Compilar e incorporar um kernel upstream compatível com a versão e a
   assinatura observadas.
2. Usar um compilador científico que produza C++ standalone.
3. Usar uma implementação semântica própria, somente quando sua conformidade
   estiver especificada e testada.

Cada combinação relevante inclui dtype, rank, eixo, `kind`, `order` e demais
constantes. Uma operação suportada para `float64[1D]` não implica suporte
automático para arrays arbitrários.

### Gate Pythran

Pythran é um candidato importante, mas não deve ser assumido como fundação sem
um spike. Sua documentação oficial descreve geração de C++ templado sem glue
Python (`-e`), containers Python e suporte parcial amplo a NumPy, inclusive
`argsort`:

- <https://pythran.readthedocs.io/en/latest/MANUAL.html#getting-pure-c>
- <https://pythran.readthedocs.io/en/latest/SUPPORT.html>

O gate precisa provar no WSL:

- geração a partir de uma função normalizada sem classes;
- chamada direta pelo adaptador RKO;
- ausência de dependência runtime de CPython/NumPy no executável;
- equivalência de `argsort`, containers, `round`, slices e erros relevantes;
- reentrância em avaliações concorrentes;
- tamanho do binário e tempo de compilação aceitáveis;
- build reproduzível com versão do Pythran fixada.

A própria documentação avisa que o nível C++ de versões `0.x` não possui forte
garantia de compatibilidade. Portanto, se aprovado, o toolchain é fixado por
versão e fingerprint. A RKO-IR permanece independente dele.

## 9. Backend e integração C++

O módulo gerado não deve conhecer `TSol`:

```cpp
namespace rko_generated {

struct FrozenInstance { /* schema gerado */ };

double evaluate(
    std::span<const double> keys,
    const FrozenInstance& instance
);

}  // namespace rko_generated
```

Um adaptador fino implementa o contrato do RKO original:

```cpp
struct TProblemData {
    int n{};
    rko_generated::FrozenInstance instance;
};

double Decoder(TSol& s, const TProblemData& data) {
    return rko_generated::evaluate(
        std::span<const double>{s.rk.data(), s.rk.size()},
        data.instance
    );
}
```

O código gerado não modifica `s.rk`, não acessa outros campos de `TSol` e não
mantém `static` mutável.

O build deve consumir um `BuildPlan` explícito e content-addressed. Não deve
copiar o programa inteiro, substituir texto em headers nem interpretar stdout
humano com regex. O executável precisa emitir resultado estruturado contendo:

- chaves com `max_digits10`;
- objetivo;
- fingerprint do código e da instância;
- status e tempos.

## 10. Diagnóstico de capacidade

O diagnóstico é uma API versionada, não uma mensagem livre. Exemplo:

```text
code: RKO-COMP-NUMPY-004
span: environment.py:71:18
call_chain:
  decoder
  -> self.rank_jobs
  -> numpy.argsort
resolved_target: numpy.argsort
signature: float64[2D], axis=<runtime>
reason: axis dinâmico não suportado pelo provider selecionado
adaptation: usar um eixo constante ou separar a operação do hot path
```

O relatório completo contém:

- raízes e grafo alcançável;
- tipos inferidos e incertezas;
- campos capturados no schema;
- especializações realizadas;
- efeitos encontrados;
- providers selecionados;
- operações suportadas e bloqueadas;
- branches eliminados;
- resultado dos testes diferenciais.

## 11. Validação

Uma construção só é declarada suportada quando possui:

1. regra de resolução;
2. regra de tipos;
3. semântica na IR;
4. interpretação de referência;
5. lowering/backend;
6. teste diferencial positivo;
7. teste negativo e diagnóstico.

Camadas de teste:

- unitários do resolver, tipos, efeitos e verificador;
- golden tests da IR e do schema;
- equivalência Python original versus interpretador IR;
- equivalência direta de resultados de cada provider;
- equivalência interpretador IR versus C++;
- casos adversariais: vazio, limites, índices negativos, empates, `NaN`,
  infinidades, zero com sinal e overflow;
- teste `A, B, A` para detectar estado persistente;
- harness concorrente isolado para reentrância;
- RKO C++ inicialmente com uma thread;
- RKO completo paralelo somente depois da validação isolada.

KP, TSP, NumPy-TSP e Ball são corpus de regressão. Nenhum deles pode gerar uma
condição especial no compilador.

Cobertura futura deve ser priorizada por um scanner sobre uma coleção maior de
ambientes reais, medindo nós AST, chamadas, dados e fronteiras nativas.

## 12. Estrutura proposta

```text
src/rko/compiler/
|-- api.py
|-- diagnostics.py
|-- frontend/
|   |-- source_loader.py
|   |-- symbols.py
|   |-- call_graph.py
|   |-- closure_conversion.py
|   `-- specialization.py
|-- types/
|   |-- model.py
|   |-- constraints.py
|   `-- inference.py
|-- effects/
|   `-- analysis.py
|-- ir/
|   |-- model.py
|   |-- verifier.py
|   |-- interpreter.py
|   `-- serialization.py
|-- snapshot/
|   |-- schema.py
|   |-- extractor.py
|   `-- codec.py
|-- providers/
|   |-- protocol.py
|   |-- registry.py
|   |-- builtins.py
|   |-- math.py
|   `-- numpy/
|-- backends/
|   |-- cpp/
|   `-- pythran/
|-- integrations/rko_cpp/
|   |-- adapter.py
|   |-- build_plan.py
|   |-- result_protocol.py
|   `-- wsl_toolchain.py
`-- validation/
    |-- differential.py
    |-- concurrency.py
    `-- provider_contracts.py
```

## 13. Roadmap com gates

### Marco 0 — congelar o spike

- Manter o compilador atual apenas como prova e suíte de aceitação.
- Registrar divergências semânticas conhecidas.
- Não adicionar novos `if` de AST ou bibliotecas ao `FunctionLowerer` atual.

Gate: KP, TSP e NumPy-TSP continuam reproduzíveis no WSL.

### Marco 1 — análise generalista, sem C++

- Novo source loader e resolvedor recursivo.
- Grafo verdadeiro de métodos, funções, closures e bibliotecas.
- `CapabilityReport` estruturado.
- Scanner de corpus.

Gate: Ball e exemplos produzem grafos reais; toda falha aponta linha, cadeia e
alvo resolvido.

### Marco 2 — tipos, efeitos e dados

- Sistema de tipos imutável e constraint solver.
- `InstanceSchema` e snapshot versionados.
- Classificação `FrozenInput`/`EvaluationLocal`/estado proibido.
- Especialização de `final_solution=False`.

Gate: schemas são reutilizáveis entre instâncias compatíveis e nenhuma mutação
compartilhada passa silenciosamente.

### Marco 3 — IR verificável

- IR estruturada com semântica Python explícita.
- Verificador e interpretador de referência.
- Builtins fundamentais, containers e controle de fluxo.

Gate: KP, TSP e o caminho compilável do Ball coincidem com o Python no
interpretador, antes de gerar C++.

### Marco 4 — decisão de backend

- Spike Pythran standalone no WSL.
- Reescrita limpa do backend C++ mínimo sobre a nova IR.
- Comparação de cobertura, semântica, dependências, tamanho e build.

Gate: escolher backend primário por evidência. Pythran pode permanecer provider
opcional; a arquitetura não depende da escolha.

### Marco 5 — integração nativa

- `FrozenInstance + evaluate()`.
- Adaptador fino para `Problem.h`.
- Build content-addressed e protocolo de resultado estruturado.
- Chaves em precisão completa.

Gate: equivalência IR/C++ no WSL e reconstrução final correta no Python.

### Marco 6 — providers e cobertura

- Builtins e `math` prioritários.
- NumPy por assinaturas observadas no corpus.
- Compilação recursiva de helpers Python.
- Suítes de conformidade por provider.

Gate: cada novo ambiente ou compila integralmente, ou recebe relatório de
adaptação preciso; nenhum caso especial pelo nome do ambiente.

## Primeiro incremento implementável

O próximo código novo deve ser somente o Marco 1:

```python
report = analyze_environment(env)
```

Ele deve retornar um grafo resolvido e um relatório estruturado, sem gerar C++.
Isso substitui a lista manual de métodos usada no estudo Ball e estabelece a
fundação sobre a qual tipos, efeitos e backends poderão crescer sem gambiarras.

## Regras que evitam gambiarras

- Nenhuma condição por classe, ambiente, instância ou nome de problema.
- Nenhuma chamada reconhecida apenas por string quando o binding pode ser
  resolvido.
- Nenhuma aproximação semântica silenciosa.
- Nenhuma biblioteca escondida dentro do visitante AST.
- Nenhum dado mutável compartilhado no decoder nativo.
- Nenhuma geração C++ antes de a IR ser verificada e interpretada.
- Nenhum provider sem versão, assinatura, efeitos e teste de conformidade.
- Nenhum sucesso baseado somente no custo de poucas amostras.
- Nenhum patch textual do RKO C++ no pipeline de produção.

