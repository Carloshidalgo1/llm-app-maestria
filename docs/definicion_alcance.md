# Definición del alcance del sistema Q&A

## Proyecto

Construcción de una base de conocimiento semántico y un sistema básico de preguntas y respuestas para **Alimentos Cárnicos S.A.S.**, filial del **Grupo Nutresa** perteneciente al Negocio Cárnico de Colombia.

El alcance se define a partir de la información pública recopilada en la carpeta `data/processed/dataset_carnicos`, con énfasis en la interacción inicial de un cliente, consumidor o interesado que desea conocer información básica, comercial, institucional y de servicio sobre la empresa.

## Propósito del sistema

El sistema Q&A debe permitir que un usuario formule preguntas en lenguaje natural y reciba respuestas claras, breves y verificables sobre Alimentos Cárnicos S.A.S., usando únicamente el contenido disponible en la base de conocimiento construida con web scraping y documentos públicos.

El objetivo no es crear todavía un chatbot transaccional ni un sistema RAG completo, sino validar que el modelo pueda responder preguntas frecuentes y de primer contacto con base en el contexto consolidado en el prompt del sistema.

## Usuarios objetivo

El sistema está dirigido principalmente a:

- Consumidores que desean conocer marcas, productos, presentaciones, puntos de venta y canales de atención.
- Clientes institucionales o mayoristas interesados en productos cárnicos, carnes frescas, distribución o contacto comercial.
- Personas interesadas en información general sobre la empresa, su historia, sedes, relación con Grupo Nutresa y presencia en el Valle del Cauca.
- Usuarios que necesitan orientación básica sobre PQRS, tratamiento de datos personales, solicitudes de empleo, donaciones, patrocinios, degustaciones o visitas a planta.
- Personas que consultan compromisos de sostenibilidad, bienestar animal, abastecimiento responsable y gestión con la comunidad.

## Temas incluidos en el alcance

### 1. Información institucional

El asistente debe poder responder preguntas sobre:

- Razón social: Alimentos Cárnicos S.A.S.
- Relación con Grupo Nutresa y el Negocio Cárnico.
- Historia y origen de la empresa, incluyendo la referencia a Salsamentaría Suiza y Rica Rondo.
- Actividad económica principal: procesamiento y conservación de carne y productos cárnicos.
- Domicilio principal en Yumbo, Valle del Cauca.
- Sitio web corporativo y canales digitales identificados.
- Certificaciones y lineamientos generales de calidad reportados en el dataset.

### 2. Marcas y portafolio de productos

El sistema debe cubrir consultas generales sobre las marcas y categorías presentes en la base de conocimiento:

- Marcas asociadas al portafolio: Rica / Rica Rondo, Cunit, Suizo, Americana y Pietrán.
- Carnes frías procesadas y embutidos: salchichas, jamones, mortadelas, salchichones, chorizos, longanizas y butifarras.
- Carnes frescas y cortes especiales: cortes de res, cortes del cuarto delantero, cortes de la rueda, recortes y cortes finos.
- Productos premium o madurados, carnes curadas, tocineta, salamis y productos ahumados.
- Productos enlatados, de larga vida, platos listos y pasabocas.
- Presentaciones y gramajes de productos Rica y Cunit cuando estén documentados en `informacion_basica.md`.

### 3. Canales de contacto y atención al cliente

El asistente debe orientar al usuario sobre:

- Dirección de la sede principal: Carrera 40 # 12A - 13, Yumbo, Valle del Cauca.
- Teléfonos y líneas de atención reportadas en el dataset.
- Sitio web corporativo y sitios de marca.
- Puntos de venta propios documentados en Cali, Bogotá y Barranquilla.
- Ruta general para radicar PQRS.
- Canales para ventas al por mayor, distribuidores autorizados y clientes institucionales.
- Horarios de atención cuando estén disponibles en la base de conocimiento.

Cuando el dato provenga de directorios o pueda cambiar, el asistente debe recomendar verificarlo en los canales oficiales de la empresa.

### 4. Procesos básicos para usuarios

El sistema debe responder preguntas de orientación inicial sobre:

- Cómo presentar una PQRS.
- Cómo solicitar degustaciones.
- Cómo se gestionan donaciones y por qué no se entregan directamente.
- Cómo aplicar a ofertas de empleo o prácticas empresariales.
- Cómo solicitar patrocinios.
- Cómo solicitar visitas a planta y requisitos básicos.
- Derechos del titular de datos personales según la política de Habeas Data.
- Uso de información personal de visitantes del sitio web según la política de uso.

### 5. Operación regional y cobertura

El sistema debe poder explicar:

- Presencia operativa de la empresa en el Valle del Cauca.
- Importancia de la sede de Yumbo.
- Presencia comercial y de distribución en Cali.
- Relación con Comercial Nutresa S.A.S. como apoyo a la distribución.
- Información pública sobre la operación cerrada en Puerto Tejada, únicamente como hecho documentado en la base y sin emitir juicios.

### 6. Sostenibilidad y responsabilidad corporativa

El asistente debe cubrir preguntas generales sobre:

- Compromisos de sostenibilidad del Grupo Nutresa aplicados al Negocio Cárnico.
- Bienestar animal y abastecimiento responsable.
- Compromiso de producción propia de cerdas en gestación libre de jaulas para 2027.
- Restricciones frente a clonación, modificación genética y promotores de crecimiento.
- Control de medicamentos, antibióticos y tiempos de retiro.
- Gestión con la comunidad: nutrición, voluntariado, apoyo al arte, cultura y programas sociales.
- Iniciativas ambientales y empaques cuando estén presentes en la base.

### 7. Noticias, datos financieros y asuntos públicos

El sistema puede responder consultas generales sobre:

- Datos financieros recientes incluidos en el dataset.
- Hechos relevantes de 2024 y 2025 documentados en la base.
- Reconocimientos de sostenibilidad.
- Litigios o asuntos legales públicos mencionados en `informacion_basica.md`, siempre con lenguaje neutral y limitado a lo documentado.

## Temas fuera del alcance

El sistema no debe responder como si tuviera acceso a información en tiempo real ni a sistemas internos de la compañía. Quedan fuera del alcance:

- Precios actualizados, promociones vigentes, inventario o disponibilidad en tiendas.
- Cotizaciones formales, pedidos, facturación, pagos, entregas o seguimiento de compras.
- Radicación real de PQRS o modificación de datos personales.
- Estado de procesos de selección, hojas de vida, prácticas o vacantes específicas.
- Diagnósticos médicos, nutricionales o recomendaciones de salud personalizadas.
- Asesoría legal, fiscal, laboral o financiera.
- Información confidencial, interna o no publicada por la empresa.
- Confirmación de datos no presentes en `data/processed/dataset_carnicos`.
- Opiniones, acusaciones o conclusiones no respaldadas por el material recopilado.
- Preguntas sobre empresas, marcas o productos no relacionados con Alimentos Cárnicos S.A.S. o Grupo Nutresa.

## Reglas de respuesta del asistente

- Responder únicamente con base en el contexto consolidado desde `data/processed/dataset_carnicos`.
- Si la información no está disponible, indicarlo de forma explícita y sugerir consultar los canales oficiales.
- No inventar teléfonos, direcciones, marcas, horarios, precios ni procesos.
- Mantener un tono claro, formal y orientado al usuario.
- Priorizar respuestas cortas para preguntas simples y respuestas estructuradas para procesos paso a paso.
- Diferenciar entre información corporativa, información comercial y orientación de servicio al cliente.
- Advertir cuando un dato pueda requerir verificación porque puede cambiar con el tiempo.

## Fuentes principales del dataset

La definición de alcance se basa principalmente en los siguientes archivos:

- `data/processed/dataset_carnicos/informacion_basica.md`
- `data/processed/dataset_carnicos/carnes-frescas.md`
- `data/processed/dataset_carnicos/preguntas-frecuentes.md`
- `data/processed/dataset_carnicos/contacto.md`
- `data/processed/dataset_carnicos/nuestro-compromiso.md`
- `data/processed/dataset_carnicos/bienestar-animal.md`
- `data/processed/dataset_carnicos/politica-de-uso.md`
- `data/processed/dataset_carnicos/derechos-del-consumidor.md`
- `data/processed/dataset_carnicos/nuestra-historia.md`
- `data/processed/dataset_carnicos/condiciones-de-uso.md`

## Preguntas de validación propuestas

Estas preguntas pueden usarse como base para las pruebas exigidas en el módulo:

1. ¿Qué empresa es Alimentos Cárnicos S.A.S.?
2. ¿A qué grupo empresarial pertenece Alimentos Cárnicos?
3. ¿Dónde está ubicada la sede principal de la empresa?
4. ¿Cuál es la historia de Alimentos Cárnicos?
5. ¿Qué relación tiene Rica Rondo con Alimentos Cárnicos?
6. ¿Qué marcas hacen parte del portafolio mencionado en la base de conocimiento?
7. ¿Qué productos ofrece la marca Cunit?
8. ¿Qué productos ofrece la marca Rica?
9. ¿Qué tipos de carnes frescas comercializa la empresa?
10. ¿Qué cortes de res aparecen en el portafolio de carnes frescas?
11. ¿Cómo puedo comunicarme con servicio al cliente?
12. ¿Cómo puedo radicar una PQRS?
13. ¿Cuáles son los puntos de venta propios mencionados?
14. ¿Cómo puede un cliente solicitar una degustación?
15. ¿La empresa realiza donaciones directamente?
16. ¿Cómo puedo aplicar a una oferta de empleo o práctica?
17. ¿Qué requisitos existen para solicitar una visita a planta?
18. ¿Qué compromisos tiene la empresa sobre bienestar animal?
19. ¿Qué acciones sociales o comunitarias aparecen en el dataset?
20. ¿Qué derechos tengo sobre mis datos personales según la política de Habeas Data?

## Criterios de aceptación

El alcance se considera cubierto si el sistema:

- Responde correctamente las preguntas de validación usando la información del dataset.
- Rechaza o redirige preguntas fuera del alcance sin inventar datos.
- Mantiene coherencia entre respuestas relacionadas con historia, marcas, productos y contacto.
- Indica limitaciones cuando la información pueda estar desactualizada o no esté disponible.
- Permite demostrar en Streamlit o Gradio una interacción básica de Q&A para un primer contacto con la empresa.

