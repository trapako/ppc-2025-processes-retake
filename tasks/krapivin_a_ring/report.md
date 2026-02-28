# Топология Кольцо

- Студент: Крапивин Александр Сергеевич, группа 3823Б1ПР1
- Технология: SEQ, MPI
- Вариант: 7

## 1. Введение

Задачей является реализация передачи сообщения по кольовой топологии, выполненная в последовательном и MPI виде передачи сообщения от источника до цели по кольцу процессов. Также необходимо зафиксировать пройденный путь и проанализировать поведение алгоритма по времени.

## 2. Постановка задачи
Для заданных параметров входа (номер источника source_rank, номер цели target_rank и некоторое целевое целочисленное значение data) необходимо передать сообщение последовательно по кольцу процессов от source до target, при этом каждый процесс–участник добавляет свою метку (номер ранга) в историю пути path_history. В конце выполнение задачи должно вернуть вектор пройденных рангов (path).


Входные данные: структура { source_rank: int, target_rank: int, data: int }.
Выходные данные: вектор std::vector<int> - последовательность рангов, через которые прошло сообщение.

## 3. Базовый алгоритм (Последовательный)
```cpp
std::vector<int> path;
int current = source;
path.push_back(current);
while (current != target) {
  current = (current + 1) % size;
  path.push_back(current);
}
```
### Описание алгоритма:
Последовательная версия вычисляет количество процессов (size), приводит номера source и target к диапазону [0, size-1] и формирует путь по кольцу от source до target инкрементируя индекс по модулю size.
Для имитации вычислительной нагрузки в цикле выполняется набор операций в течение фиксированного времени 800 ms.

## 4. Схема распараллеливания
- Используется кольцевая топология MPI каждый процесс знает prev и next.
- Только участники, попадающие в интервал от source до target по кольцу, участвуют в пересылке сообщения и сформировании пути включая source и target.
- source инициализирует path_history включает свой ранг, выполняет задержку и отправляет размер пути, сами элементы пути и сопутствующие данные следующему процессу.
- Каждый участник получает пакет от предыдущего, добавляет свой ранг в историю, выполняет задержку и либо завершает, либо пересылает дальше.

### Коммуникационные схема:
- MPI_Send / MPI_Recv - отправка размера и массива пути, а также сопутствующего data.
- MPI_Barrier и MPI_Comm_dup используются для синхронизации и изоляции коммуникаций.
Особенности:
- Передача представляет собой последовательную цепочку зависимых передач: следующий шаг ожидает данных от предыдущего.

## 5. Детали реализации

### Структура кода
Файлы:
- `common/include/common.hpp` - общие типы данных
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` - SEQ реализация
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` - MPI реализация
- Тесты в папках `tests/functional/` и `tests/performance/`

Ключевые методы:
- ComputeIsParticipant() - проверяет, участвует ли ранк в пути между source и target.
- HandleSource() - обработка на стороне source: инициализация path_history, задержка, отправка пакета следующему.
- HandleParticipant() - обработка у промежуточного участника: приём, задержка, добавление в path_history, пересылка дальше или завершение.

### Особенности реализации
- Все коммуникации выполняются точечно (point-to-point), полный путь передаётся последовательно.
- Для симуляции затрат на обработку использована небольшая вычислительная задержка AddDelay() 200 ms в MPI-реализации и более длительная в SEQ 800 ms.
- Функции защищают от неверных размеров и некорректных значений path_size при приёме.

## 6. Экспериментальная установка
### Оборудование и ПО
- **Процессор:** AMD Ryzen 7 4800H (2.9 Ггц)
- **ОС:** Ubuntu devcontainer (host Windows)
- **Компилятор:** gcc
- **Тип сборки:** release

### Тестовые сценарии
- Тестовые сценарии задают ring_size через mpirun -n N.

## 7. Результаты и обсуждение

### 7.1 Проверка корректности
- Функциональные тесты проходят успешно: GetOutput() возвращает корректную последовательность рангов от source до target по модулю размера кольца.
- Пограничные случаи: source == target, source > target склейка по модулю и случаи с world_size == 1 корректно обработаны.

### 7.2 Производительность

| Процессы  | Время, c   | 
|-----------|------------|
| 1 (SEQ)   | 0.8016     | 
| 4         | 0.8013     |
| 30        | 6.5685     |


## 8. Выводы
- Реализована корректная схема последовательной передачи по кольцу: сообщение проходит по всем ранкам от source к target с накоплением истории пути.
- MPI-реализация корректно обрабатывает границы и случаи source==target и склейки при source>target.

## 9. Источники
1. Курс лекций по параллельному программированию Сысоева Александра Владимировича. 
2. Документация по курсу: https://learning-process.github.io/parallel_programming_course/ru

## Приложение
```cpp
bool KrapivinARingMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input = GetInput();
  int source = input.source_rank;
  int target = input.target_rank;

  if (world_size > 0) {
    source = source % world_size;
    target = target % world_size;
  }

  MPI_Group world_group = MPI_GROUP_NULL;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Comm ring_comm = MPI_COMM_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &ring_comm);

  int ring_rank = 0;
  int ring_size = 0;
  MPI_Comm_rank(ring_comm, &ring_rank);
  MPI_Comm_size(ring_comm, &ring_size);

  int next_rank = (ring_rank + 1) % ring_size;
  int prev_rank = (ring_rank - 1 + ring_size) % ring_size;

  MPI_Group_free(&world_group);

  bool is_participant = ComputeIsParticipant(ring_rank, source, target);

  if (ring_rank == source) {
    HandleSource(ring_comm, ring_rank, next_rank, target, input.data);
  } else if (is_participant) {
    HandleParticipant(ring_comm, prev_rank, next_rank, ring_rank, target);
  } else {
  }

  MPI_Barrier(ring_comm);
  MPI_Comm_free(&ring_comm);
  return true;
}
```