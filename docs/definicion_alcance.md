# Definición del alcance del sistema Q&A

## Proyecto

Construcción de un asistente conversacional basado en LLM para **Alimentos Cárnicos S.A.S.**, filial del **Grupo Nutresa** perteneciente al Negocio Cárnico de Colombia.

El asistente responde preguntas de primer contacto a través de **WhatsApp** (vía Twilio + N8N + FastAPI), usando una base de conocimiento construida con web scraping y documentos públicos de la empresa, indexada mediante embeddings vectoriales (RAG).

## Propósito del sistema

El sistema permite que un usuario formule preguntas en lenguaje natural por WhatsApp y reciba respuestas claras, breves y verificables sobre Alimentos Cárnicos S.A.S., usando únicamente el contenido disponible en la base de conocimiento.

Cuando el agente no encuentra evidencia documental suficiente (`confidence = "low"`), el flujo N8N escala automáticamente la consulta a un asesor humano y notifica al cliente que será atendido en breve.

## Canal de acceso

El asistente opera sobre el siguiente stack de despliegue:

- **WhatsApp** — canal de entrada y salida del usuario final
- **Twilio** — gateway SMS/WhatsApp
- **N8N** — orquestación del flujo de mensajes
- **ngrok** — exposición pública HTTPS del servidor local
- **FastAPI** — API REST del agente (`POST /chat`)
- **LangGraph ReAct + GPT-4o-mini** — motor de razonamiento y RAG

## Usuarios objetivo

- Consumidores que desean conocer marcas, productos, presentaciones, puntos de venta y canales de atención.
- Clientes institucionales o mayoristas interesados en productos cárnicos, carnes frescas, distribución o contacto comercial.
- Personas interesadas en información general sobre la empresa, su historia, sedes, relación con Grupo Nutresa y presencia en el Valle del Cauca.
- Usuarios que necesitan orientación básica sobre PQRS, tratamiento de datos personales, solicitudes de empleo, donaciones, patrocinios, degustaciones o visitas a planta.
- Personas que consultan compromisos de sostenibilidad, bienestar animal, abastecimiento responsable y gestión con la comunidad.

## Temas incluidos en el alcance

### 1. Información institucional

- Razón social: Alimentos Cárnicos S.A.S.
- Relación con Grupo Nutresa y el Negocio Cárnico.
- Historia y origen, incluyendo Salsamentaría Suiza y Rica Rondo.
- Actividad económica principal: procesamiento y conservación de carne.
- Domicilio principal en Yumbo, Valle del Cauca.
- Certificaciones y lineamientos generales de calidad reportados en el dataset.

### 2. Marcas y portafolio de productos

- Marcas: Rica / Rica Rondo, Cunit, Suizo, Americana y Pietrán.
- Carnes frías procesadas y embutidos: salchichas, jamones, mortadelas, salchichones, chorizos, longanizas y butifarras.
- Carnes frescas y cortes especiales de res.
- Productos premium, curados, tocineta, salamis y ahumados.

### 3. Canales de contacto y atención al cliente

- Dirección sede principal: Carrera 40 # 12A - 13, Yumbo, Valle del Cauca.
- Teléfonos y líneas de atención reportadas en el dataset.
- Puntos de venta propios documentados en Cali, Bogotá y Barranquilla.
- Ruta general para radicar PQRS.
- Canales para ventas al por mayor y clientes institucionales.

### 4. Procesos básicos para usuarios

- Cómo presentar una PQRS.
- Cómo solicitar degustaciones, donaciones, patrocinios y visitas a planta.
- Cómo aplicar a ofertas de empleo o prácticas.
- Derechos del titular de datos personales según Habeas Data.

### 5. Sostenibilidad y responsabilidad corporativa

- Compromisos de sostenibilidad del Grupo Nutresa aplicados al Negocio Cárnico.
- Bienestar animal: meta cage-free 2027, abastecimiento responsable.
- Restricciones frente a clonación, modificación genética y promotores de crecimiento.
- Gestión con la comunidad y programas sociales.

## Temas fuera del alcance

- Precios actualizados, promociones vigentes, inventario o disponibilidad.
- Cotizaciones, pedidos, facturación o seguimiento de compras.
- Radicación real de PQRS o modificación de datos personales.
- Estado de procesos de selección o vacantes específicas.
- Diagnósticos médicos, nutricionales o asesoría legal.
- Información confidencial o no publicada por la empresa.
- Empresas, marcas o productos no relacionados con Alimentos Cárnicos S.A.S.

## Reglas de respuesta del asistente

- Responder únicamente con base en fragmentos recuperados por RAG desde `data/processed/dataset_carnicos`.
- Si la información no está disponible, indicarlo explícitamente (`confidence = "low"`) — N8N escala al asesor.
- No inventar teléfonos, direcciones, marcas, horarios, precios ni procesos.
- Mantener tono claro, formal y orientado al usuario.
- Respuestas en formato WhatsApp: *negrita* con asterisco simple, viñetas con `•`.

---

## Preguntas de validación para prueba del flujo N8N

Las siguientes preguntas están organizadas según el resultado esperado en el campo `confidence` de la API. Úsalas para verificar que el flujo IF de N8N funciona correctamente.

### Grupo A — Deben responder con confianza alta (rama FALSE del IF → respuesta al cliente)

Estas preguntas tienen respuesta directa en la base de conocimiento:

1. ¿Qué empresa es Alimentos Cárnicos S.A.S.?
2. ¿A qué grupo empresarial pertenece Alimentos Cárnicos?
3. ¿Dónde está ubicada la sede principal de la empresa?
4. ¿Qué marcas hacen parte del portafolio de la empresa?
5. ¿Qué productos ofrece la marca Rica?
6. ¿Qué productos ofrece la marca Cunit?
7. ¿Qué tipos de carnes frescas comercializa la empresa?
8. ¿Cómo puedo comunicarme con servicio al cliente?
9. ¿Qué compromisos tiene la empresa sobre bienestar animal?
10. ¿Qué derechos tengo sobre mis datos personales según la política de Habeas Data?
11. ¿Cuál es la historia de Alimentos Cárnicos?
12. ¿Cómo puedo radicar una PQRS?
13. ¿Cuáles son los puntos de venta propios mencionados?
14. ¿Qué requisitos existen para solicitar una visita a planta?
15. ¿La empresa realiza donaciones directamente?

### Grupo B — Deben escalar al asesor (rama TRUE del IF → notificación al asesor + mensaje al cliente)

Estas preguntas están fuera del alcance documental o requieren información en tiempo real:

16. ¿Cuál es el precio actual del kilo de salchicha Rica?
17. ¿Tienen disponibilidad de producto en Bogotá esta semana?
18. ¿Cuándo es la próxima promoción de Cunit?
19. ¿Cuál es el nombre del gerente general de la empresa?
20. ¿Puedo hacer un pedido de 500 kilos de carne para el próximo lunes?

### Grupo C — Deben ser rechazadas por estar fuera del dominio

El agente debe responder que solo atiende preguntas sobre Alimentos Cárnicos S.A.S.:

21. ¿Cuál es la capital de Francia?
22. ¿Cómo se hace una pizza margarita?
23. ¿Qué opinas de la competencia de Zenú?

---

## Criterios de aceptación

El flujo se considera validado si:

| Criterio | Verificación |
|---|---|
| Grupo A llega al cliente con respuesta del agente | El cliente recibe el mensaje en WhatsApp |
| Grupo B activa el nodo IF rama TRUE | El asesor recibe la alerta y el cliente recibe confirmación |
| Grupo C es rechazado por el agente | El cliente recibe respuesta de fuera de dominio |
| El `thread_id` (número de teléfono) mantiene contexto entre turnos | Preguntas de seguimiento resuelven referencias anteriores |
| La respuesta llega en menos de 10 segundos | Medido desde que el usuario envía el mensaje hasta que llega la respuesta |
