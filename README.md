# Proyecto Integrador: Inteligencia Artificial en Ingeniería Biomédica

## 📌 Descripción General
Este repositorio contiene el desarrollo del Proyecto Integrador para la asignatura **Automatización 2** de la carrera de **Ingeniería Biomédica** en el **ITM**. El proyecto implementa soluciones avanzadas de Inteligencia Artificial (IA) y Aprendizaje Automático (Machine Learning) orientadas a la optimización de procesos clínicos y hospitalarios, divididas en dos grandes bloques temáticos: Ingeniería de Prompts/Generación de Datos Sintéticos y el desarrollo de tres Aplicaciones Biomédicas Especializadas.

---

## 📐 Estructura del Proyecto

### Bloques de Evaluación
| Bloque | Contenido | Evaluación |
| :--- | :--- | :--- |
| **Bloque 1** | Ingeniería de Prompts y Generación de Datos Sintéticos | 20 % |
| **Bloque 2** | Desarrollo y Programación de los 3 Casos Biomédicos | 20 % |

### Arquitectura de los Casos Clínicos (Bloque 2)
| Caso | Título del Proyecto | Arquitectura Tecnológica |
| :---: | :--- | :--- |
| **A** | Asistente de Triage y Protocolos Clínicos | RAG con `models/gemini-embedding-001` y candado de seguridad |
| **B** | Formulario de Equipo Médico | Fine-Tuning local de GPT-2 (`Hugging Face Trainer`) |
| **C** | Clasificador Híbrido de Triage | ML Híbrido: `StandardScaler` + `TF-IDF` + `LogisticRegression Softmax` |

---

## 🧠 Detalles de Implementación

### Bloque 1: Ingeniería de Prompts y Datos Sintéticos
Diseño metodológico de directrices estructuradas para Modelos de Lenguaje (LLMs) con el fin de generar bases de datos sintéticas balanceadas y validadas mediante scripts de Python:
* **RAG de Protocolos Clínicos:** Generación de información médica técnica y dosificaciones precisas (Adrenalina, Amiodarona, Atropina) bajo el rol de un especialista en emergencias.
* **Fine-Tuning de Reportes Técnicos:** Creación de un diccionario de alineación semántica (20 pares de entrenamiento) para transformar reportes informales de técnicos hospitalarios en documentación clínica formal bajo el estándar IEC 60601.
* **Dataset Híbrido para Triage:** Estructuración de una matriz de 100 registros balanceados que asocian variables fisiológicas (signos vitales) y descripciones sintomáticas con su respectiva especialidad médica destino.

### Bloque 2: Casos Biomédicos Desarrollados

#### Caso A: Asistente de Triage y Protocolos Clínicos (Arquitectura RAG)
Sistema de consulta a "libro abierto" para entornos de urgencias. Convierte un manual de dosificación de medicamentos críticos en vectores utilizando `models/gemini-embedding-001`. Ante la consulta del personal médico en lenguaje natural, recupera los fragmentos con mayor similitud semántica y los inyecta en un prompt con **candado de seguridad** para forzar una respuesta verídica y libre de alucinaciones.

#### Caso B: Formulario de Equipo Médico (Fine-Tuning de GPT-2)
Modelado de lenguaje especializado enfocado en la Ingeniería Clínica. Utiliza las capacidades de `Hugging Face Trainer` para realizar el ajuste fino local de un modelo GPT-2, entrenándolo para estandarizar reportes de mantenimiento biomédico, corrigiendo la jerga técnica informal hacia descripciones normativas y profesionales.

#### Caso C: Clasificador Híbrido de Triage
Modelo predictivo enfocado en la clasificación y priorización de pacientes prehospitalarios hacia 5 especialidades (Cardiología, Neurología, Traumatología, Neumología y Cirugía General). Combina analítica cuantitativa y procesamiento de texto:
1.  **Datos Numéricos:** Normalización de signos vitales (FC, PAS, PAD, FR, SpO2, Glasgow y Edad) mediante `StandardScaler`.
2.  **Datos de Texto:** Vectorización de los síntomas reportados usando `TF-IDF`.
3.  **Algoritmo:** Fusión de características alimentadas a una regresión logística multinomial (`LogisticRegression Softmax`).

---

## 🛠️ Requisitos e Instalación

El entorno requiere la verificación e instalación de las siguientes dependencias base:

```bash
pip install numpy pandas scikit-learn transformers torch google-generativeai -q
