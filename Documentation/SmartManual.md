Guía de Instalación 

# y Manual de Usuario 

## Sistema de Imagen Infrarroja Dual con carrusel x8 

Seguimiento de Trayectoria de Múltiples Gusanos 

(Algoritmo de Aprendizaje Automático) + Detección de Dispersión de Luz Infrarroja Sistema de Adquisición de Datos: WMicrotracker SMARTx8 

Versión de Hardware: SMARTx8 V1.0 

Versión de Software: SMART V2.12 (2024) 

Gracias por adquirir el sistema SMARTx8. El siguiente documento le guiará a través del proceso de instalación. 

Este producto está protegido bajo patentes internacionales: P20060105084AR, PCT/IB2007/054628, EPO y patente estadounidense concedida, propiedad del Consejo Nacional de Investigaciones Científicas y Técnicas de Argentina (CONICET) y licenciada a PHYLUMTECH S.A.; y P20190100121AR, PCT/ES2020/070029, patente EPO yestadounidense propiedad de PHYLUMTECH S.A. 

Queda estrictamente prohibida la copia de cualquier parte de este producto, total o parcialmente, y podría acarrear sanciones legales. Este producto se proporciona "TAL CUAL". No se permiten modificaciones sin permiso de PHYLUMTECH. Al adquirir este producto, usted reconoce y acepta estos términos y condiciones. 

Este producto es solo para fines de investigación y no está destinado para uso diagnóstico humano.  (©2022). H echo en Argentina. Contenidos 

## I. Acerca del SMART 

##  Componentes Incluidos 

##  Requisitos Adicionales 

##  Dimensiones del Producto y Fabricación 

## II. Guía de Instalación y Configuración 

##  Instalación del Software 

##  Configuración del Hardware 

## III . Software de  Adquisición 

##  Inicio del Software 

##  Ventana de Adquisición de Datos: Componentes de la Pantalla 

## y Operación I. Acerca del SMART 

El WMicrotracker SMART es un sistema modular diseñado para el seguimiento de pequeños organismos en formato de placa Petri de 35 mm. Permite la cuantificación confiable del movimiento de poblaciones animales y es compatible con una variedad de organismos, incluyendo C. elegans y nematodos relacionados, larvas de pez cebra y Drosophila, así como pequeños insectos. 

Tiene la capacidad de trabajar en 2 modos diferentes: 

1- Imágenes por IR boca abajo: Este modo permite el seguimiento de múltiples trayectorias de gusanos usando cultivos en NGM. Se basa en el fenómeno óptico de Amplificación de Siluetas por Refracción Infrarroja, donde las ondas de luz infrarroja se refractan en la interfaz gusano-agar, generando una imagen amplificada capturada por un sistema de cámara de alta resolución. El procesamiento digital de la imagen se realiza utilizando software diseñado especialmente para la adquisición de datos en tiempo real. 

2- Dispersión de Luz por IR : Este modo permite la cuantificación del comportamiento de múltiples organismos mayores de 0.1 mm utilizando cultivos en medios sólidos, líquidos o gaseosos. Permite definir diferentes áreas de actividad en la placa, lo cual es útil en experimentos de quimiotaxis. El método patentado detecta el movimiento a través de la dispersión de luz causada por una grilla de microhaces infrarrojos. 

Algunas características técnicas del sistema: 

 Adquisición no invasiva. 

 No se ve afectado por bacterias (aunque cultivos bacterianos muy densos o oscuros pueden afectar la precisión del sistema). 

 Compatible con RNAi y screenings de compuestos.  Permite la evaluación de múltiples protocolos utilizando animales cultivados en agar, medios de cultivo líquidos e insectos cultivados en aire. 

 El formato de cultivo de microplaca preferido para el sistema SMART son las placas Petri claras de 35 mm. Se recomienda utilizar el modelo 35mm-Greiner Bio-One #627161. NOTA: La placa Petri debe ser utilizada con la tapa puesta y se recomienda sellar la placa con film. 

## Componentes incluidos: 

## Requerimientos Adicionales: 

● Compatible con PC IBM con los siguientes requisitos mínimos: 

🀀  Procesador Pentium Core i3 o superior 

🀀  2 GB de memoria RAM 

🀀  1 puerto USB disponible para alimentar el punto de acceso WiFi. 

🀀  Sistema operativo MSWindows 7 (o superior) 

🀀  > 1 GB de espacio libre en disco duro para el almacenamiento de 

imágenes de experimentos. ● La funcionalidad óptima del sistema requiere una temperatura ambiente de operación de 10°C a 40°C con humedad por debajo del 50%, aunque las muestras biológicas pueden tener requisitos únicos de temperatura. 

● Minimice la vibración y el polvo en su área de trabajo. 

● Evite ubicar el instrumento cerca de una ventana clara o luz brillante. 

## Dimensiones del Producto y Fabricación  

> 

LWH 22 cm x 24 cm x 22 cm (8.66 pulgadas x 9 pulgadas x 8.66 pulgadas).  

> 

Tecnología de Fabricación: Impresión 3D II. Guía de Instalación y Configuración 

## Instalación del Software 

Recomendamos verificar periódicamente el sitio web de Phylumtech para actualizaciones de software. 

1. Para descargar la carpeta .zip de instalación del software, vaya a la zona de descarga de software (https://www.phylumtech.com/home/en/support/ ), haga clic derecho en el enlace y elija "guardar enlace como". 

2. Descomprima los archivos y copie la carpeta WMicrotracker_smartx8 directamente en c:\wmicrotracker_smartx8. (Evite copiar la carpeta al escritorio o a una carpeta con un nombre muy largo.) 

3. Luego siga estas instrucciones: 

a. Conecte el dispositivo de comunicación USB-Wifi. Creará una red WiFi llamada "phylumtech.com". 

b. Conecte su computadora a la red WiFi "phylumtech.com" usando la contraseña "WMicroSMART". 

c. Ejecute el archivo ejecutable WMicrotracker_smartx8_vXX.exe 

Comentarios adicionales:  

> 

Dado que el WMicrotracker SMARTx8 utiliza un protocolo de comunicación WiFi directo, a menos que tenga una LAN cableada, no tendrá acceso a Internet en su PC cuando se conecte al dispositivo. 

## Encendido del Hardware y Configuración 

1. Conecte el dispositivo de comunicación USB-Wifi. 

2. Conecte su computadora a la red WiFi "phylumtech.com" utilizando la contraseña "WMicroSMART". 3. Conecte la fuente de alimentación (de 9VDC con salida de 1.5 Amperios) a un tomacorriente regular y el cable al conector en la parte posterior de su SMART. Luego, presione el botón ubicado en la parte posterior del dispositivo. 

4. La pantalla se encenderá y mostrará la contraseña de red configurada mientras intenta conectarse a la red WiFi "phylumtech.com" (espere unos segundos). 

Si el equipo no logra conectarse después de 1 minuto, verifique la disponibilidad de la red "phylumtech.com" y reinicie el equipo desconectándolo durante unos segundos. 

5. Después de una conexión exitosa, la pantalla mostrará la dirección IP asignada en la red Wi-Fi, la cual se utiliza para enlazar el software de adquisición con el dispositivo. 

Nota: Este número aleatorio es asignado automáticamente por el Punto de Acceso (router WiFi) cada vez que se establece la conexión. III. Software de Adquisición 

## Inicio del Software 

1. Abra la carpeta donde instaló el software y ejecute el archivo ejecutable 

"WMicroSmart". La aplicación debería iniciarse de inmediato y mostrar la "Ventana de Inicio". 

2.  Haga clic en "Iniciar Nueva Adquisición" para conectar su dispositivo SMART.  A

continuación, aparecerá la siguiente ventana: 3. Ingrese el ID del dispositivo con la IP indicada en la pantalla del dispositivo, por 

ejemplo: 

4. Haga clic en "Conectar". Si la conexión es exitosa, accederá a la ventana de adquisición de datos. Ventana de Adquisición de Datos: Componentes de la Pantalla y Operación 

1. La imagen de la placa y el entrenamiento se muestra al configurar el dispositivo IP en el software. 

a. Estación: 

Puede seleccionar la posición deseada del carrusel haciendo clic en cada número. El dispositivo rotará automáticamente y alineará esa estación frente a la cámara. 

Si desea girar manualmente el carrusel, por favor haga clic primero en el botón "parar" para permitir el movimiento libre. 

b. MODOS: 

i. a. Modo de Seguimiento para C. elegans y nematodos similares cultivados en agar: 

En este modo, el sistema utiliza la propiedad óptica de la transición de fase de la luz (aire-gusano-agar) para amplificar las siluetas de los gusanos cuando están expuestos a luz infrarroja. Un fenómeno similar ha sido descrito en aplicaciones FIM (A Multi-Purpose Worm Tracker Based on FIM | bioRxiv ). Imágenes de alta resolución son procesadas utilizando algoritmos de aprendizaje automático para identificar gusanos individuales y seguir sus trayectorias. 

La placa Petri debe colocarse boca abajo (con la tapa hacia abajo y el adaptador de placa adecuado) para la detección de gusanos cultivados en NGM. Solo es compatible con gusanos en etapa adulta de C. elegans y de tamaño similar. i. b. Modo de Seguimiento para pequeños animales acuáticos e insectos: 

La placa Petri debe colocarse boca arriba (con la tapa hacia arriba y el adaptador de placa adecuado) para la detección de organismos mayores de 1 mm. 

ii. Modo “Microbeam": El adaptador de placa con microrejilla posee una rejilla compuesta por más de 2,000 microorificios de 100 μm de ancho. La detección de actividad para este modo se basa en determinar la fluctuación de la luz de microhaz de cuadro a cuadro utilizando un algoritmo de sustracción de imagen píxel a píxel. Si la diferencia entre píxeles vecinos es mayor que un umbral, se incrementa un acumulador de actividad. Este modo de cálculo utiliza imágenes de video de baja resolución sin detección de gusanos individuales. 

La placa Petri debe colocarse boca arriba (con la tapa hacia arriba) con el adaptador de placa con microrejilla. 

c. Foco y posición de la placa: 

i. Las flechas permiten ajustar la posición del área de registro (círculo rojo) de su placa Petri. Utilice las flechas "Arriba", "Abajo", "Derecha >", y "Izquierda <" para mover el círculo rojo a la ubicación deseada. 

ii. "Diámetro": Utilice esta opción para aumentar o disminuir el tamaño del área de registro (círculo rojo) de su placa Petri. Es importante colocar el círculo rojo en el borde de la placa base de 35 mm. 

iii. "Tamaño (mm)": Esta opción le permite especificar el diámetro real del círculo rojo en milímetros. Sirve como indicador de escala para calcular distancias con precisión. 

iv. Foco: Esta opción le permite capturar imágenes en tiempo real para ajustar el enfoque del objetivo utilizando la rueda de enfoque en la parte posterior del dispositivo SMART. El tiempo designado para ajustar el enfoque está configurado en 20 segundos. Si no se logra el enfoque dentro de ese tiempo, puede presionar nuevamente el botón de enfoque. 

d. Detección de Siluetas (solo habilitada para el modo de seguimiento): 

i. "Nuevo": Permite al software ser entrenado para reconocer siluetas. Para seleccionar las siluetas de los organismos en la imagen, haga clic en la partícula con el botón izquierdo del mouse (seleccione de 5 a 10 organismos representativos). Por ejemplo; 

Tamaño del recuadro: Aumente o disminuya el tamaño del recuadro para que todo el microorganismo quepa dentro. El tamaño del recuadro de referencia para C. elegans es de 35 a 45 píxeles, dependiendo de la etapa del gusano. Luego, presione el botón "APLICAR", y las sombras de los animales se colorearan de rojo. Finalmente, presione "OK" para guardar los parámetros de entrenamiento. 

ii. "Prueba de entrenamiento": Permite verificar el rendimiento de los patrones de reconocimiento de siluetas grabados. 

e. Imagen de la placa: 

i. El WMicrotracker SMART está actualmente validado para trabajar con formato de 35 mm. La imagen mostrada es una foto de la placa tomada durante la última conexión con el dispositivo. Puede ver la placa en tiempo real al presionar el botón de foco. 2. La pantalla de adquisición de imágenes se muestra cuando selecciona el botón "Siguiente>" en la primer pantalla de configuración. 

a. Barra de Estado: 

Muestra el progreso del tiempo de ejecución como una barra amarilla. 

b. Menú de Carpeta de Proyecto: 

i. Puede crear y nombrar una nueva carpeta de proyecto escribiendo en el cuadro. 

ii. Puede acceder a un nombre de proyecto anterior mostrando el cuadro. 

La carpeta de proyecto se guardará dentro de la carpeta de instalación de su software. 

c.  Adquisición de Datos: 

i. "Read": use este botón para establecer el tiempo total de adquisición de la placa 

Petri,  en minutos. Puede elegir entre 1 y 5 minutos. 

ii. "Estímulo de Luz": puede seleccionar si desea adquirir datos en presencia de luz azul o no, o dar un pulso de luz de 10 segundos antes de la adquisición. Si selecciona esta opción, la luz azul se encenderá cuando comience la adquisición y se apagará automáticamente cuando se detenga. NOTA: No todos los dispositivos tienen luz azul incorporada. 

iii. Ciclos: para utilizar ciclos de adquisición, seleccione la opción 'Ciclos'. Esto le pedirá que especifique el número de veces que desea leer la misma placa y el intervalo de tiempo entre el inicio de cada ciclo de adquisición (Tiempo de Ciclo). La duración total del ciclo se estima en función de estos ajustes. RToff: Debido a ciertas características de MS Windows, realizar múltiples adquisiciones y procesar imágenes en tiempo real puede provocar bloqueos más frecuentes del equipo. Para mitigar este riesgo, recomendamos realizar adquisiciones sin procesamiento en tiempo real (seleccionando "RToff") y realizar el procesamiento por lotes una vez que el equipo haya completado las lecturas. 

d. Nombres de Placas: 

a. Haga doble clic en cada estación para renombrarla. 

b. Haga clic derecho en cualquier estación para abrir un menú donde puede elegir habilitarla o deshabilitarla para la adquisición. Si una estación no tiene una placa, deshabilitarla asegura que el dispositivo no adquirirá datos de ella y pasará automáticamente a la siguiente estación habilitada. INICIO de la adquisición: 

Una vez que haya configurado los ajustes de adquisición, haga clic en el botón INICIAR para comenzar el proceso de adquisición. Aparecerá una ventana emergente pidiéndole que ingrese un nombre para su experimento. Se recomienda elegir un nombre que le ayude a identificar sus datos en el futuro. 

Durante la adquisición, el software mostrará la progresión de los resultados y las imágenes en tiempo real. 

Dependiendo del modo que haya seleccionado inicialmente, ya sea "SEGUIMIENTO" o "MICROHAZ", el modo de registro y la información mostrada en el informe variarán. 

## ⇨ MODO T RACKING 

El sistema comenzará a capturar 1 imagen por segundo y procesará cada imagen en tiempo real para detectar y seguir organismos individuales dentro de la placa de Petri. RESULTADOS 

Los archivos de informe del experimento se pueden acceder fácilmente ya sea inmediatamente después de una adquisición o en un momento posterior desde el menú de resultados. 

Archivo de informe [report.csv]: 

El archivo de informe contiene los resultados de cuantificación agrupados por bloques de tiempo definidos por el usuario. Primero mostrará los resultados promedio de la población, seguidos de los resultados individuales de las partículas en la parte inferior. 

1) Resultados promedio de la población: 

- Partículas: el número promedio de partículas detectadas por cuadro. 

- Velocidad de partícula [mm/s]: la velocidad promedio de todas las partículas detalladas. 

- Distancia recorrida [mm/partícula]: la suma de la distancia recorrida por todas las partículas detectadas dividida por el número promedio de partículas detectadas por cuadro. 

- Puntuación de movilidad: la fracción de eventos de detección 

correspondientes a partículas en movimiento (distancia > 1 mm).  El ran go es 

de 0 a 1. 

- Índice de rotación: el promedio del índice de rotación de todas las partículas detalladas. 

Consulte el Anexo 1 para ver un ejemplo. 

2)  Resultados de partículas individuales: 

- Lista de resultados para cada partícula individual detectada durante al menos 10 fotogramas. 

- ID de la partícula: el número de identificación asignado por orden de aparición/reconocimiento. 

- Distancia: la distancia recorrida por esa partícula. 

- Fotogramas: el número de fotogramas en los cuales se detectó esa partícula. El dispositivo adquiere 1 fotograma por segundo. - Velocidad: la distancia recorrida por esa partícula dividida por el número de fotogramas (1 fotograma por segundo) en los cuales se detectó esa partícula. 

- Rotación: un índice que representa el cambio en la rotación de las formas corporales de los animales. Es útil para experimentos de duración de vida o animales que permanecen en su lugar haciendo pequeños movimientos, pero no recorridos largos. (Rango: 0-16). 

Consulte el Anexo 1 para obtener más información. 

El anexo 2 detalla los archivos adicionales de salida. 

## ⇨ IR Microbeam Ligth Scattering MODE 

Ejemplo de captura de microbeam: La detección de actividad para este modo se basa en determinar la fluctuación de las luces de microhaz de cuadro a cuadro mediante el uso de un algoritmo de resta de imágenes píxel a píxel. 

Ejemplo de actividad de la población de gusanos detectada en una placa Petri de 35 mm con agar NGM: 

Usando este método es posible cuantificar la actividad locomotora de la población de gusanos cultivados en medio líquido o de agar. Este modo de adquisición también proporciona información estadística sobre la ubicación de la población de gusanos, lo que lo hace útil para experimentos de quimiotaxis. Para obtener más información y la aplicación de este método de adquisición, consulte el MANUAL DE USUARIO del WMicrotracker ARENA. MENU <CHECK RESULTS> 

Para acceder a los datos de adquisiciones anteriores, navegue hasta el menú "Ver Resultados" en la ventana "Inicio". Desde allí, podrá ver sus resultados, generar informes e incluso volver a procesar su experimento entrenando nuevamente al software para reconocer siluetas. 

## Reanálisis de Placa: Componentes de pantalla y Operación. 

Secuencia del procedimiento: 

1. Seleccionar Proyecto y Adquisición 

2. Graficar Resultados 

3. Exportar Informe 4. Reprocesar datos si desea reanalizar las trayectorias de los gusanos usando un nuevo entrenamiento. 

Descripción de los botones: 

a. Proyecto: Muestra el nombre del proyecto-adquisición que está cargado. Haga clic en el 

botón ((…)_ ) para expandir el directorio donde puede buscar el experimento de interés. 

Los experimentos están organizados en carpetas por nombre de proyecto, y dentro de cada una están todas las adquisiciones listadas por fecha. Seleccione y haga doble clic para cargar el experimento deseado. 

b. Imagen de la Placa de Petri: Muestra la foto inicial de la adquisición, y durante "graficar resultados" muestra el recorrido de las partículas detectadas.  

> 

Haga clic en "Exportar BMP" para exportar la imagen del recorrido de las partículas como una imagen. 

c. Resultados:  

> 

Graficar resultados: haga clic en este botón para cargar todas las imágenes de la adquisición. Los recorridos de las partículas detectadas según el último entrenamiento guardado aparecerán en el área de diseño de la placa.  

> 

Resultados: muestra los resultados promedio de la población:  

> o

#Partículas detectadas (promedio por cuadro)  

> o

Velocidad de las partículas [mm/s]  

> o

Distancia recorrida [mm/partícula]  

> o

Índice de rotación  

> o

Histograma de velocidad [mm/s] 

d. Informe:  

> 

Tamaño de intervalo (min): elija el bloque de tiempo para el cálculo de los parámetros.  

> 

Espaciador: seleccione el mejor "espaciador" para sus necesidades de análisis de datos.  

> 

Informe: haga clic en este botón para generar un archivo .csv con los resultados promedio de la población y de cada partícula individual. 

e. Reprocesar: haga clic en este botón para redirigir a una ventana donde puede cambiar la posición de la placa o volver a entrenar el software para la detección de siluetas. El menú/opciones son similares a las de la ventana de adquisición. Presione el botón "Reprocesar Imágenes" para aplicar los cambios. El nuevo entrenamiento sobrescribirá al anterior. ANNEX 1 

Details on report average population results: 

- #Particles detected (average number per frame). 

- Average speed [mm/s]. 

- Travelled distance [mm/particle]. 

- Motility score. 

- Rotation Index. 

Calculation of Average population results: Calculation of Rotation index: 

For each detected worm, its silhouette is converted into a binary matrix with a resolution of 4 pixels by 4 pixels, where a value of 1 indicates the presence of the organism. The software then calculates the bitwise exclusive OR (XOR) between the binary matrices of successive frames. The average of the XOR values for all worms corresponds to the reported Rotation Index, which quantifies the amount of rotational movement of the worms over time. ANNEX 2 

File recording description: 

Additional files: 

The following raw data files will be located into project acquisition folder: 

\img folder 

contains all images captured during the acquisition lapse. This images can be used to generate a .GIF or .AVI video using ImageJ software 

Images are enumerated by System_ID + Frame Timing (in seconds). 

Each image file size is about 50 to 100 kbytes 

\bmp folder 

it contains the exported images for incremental worm trails Worm_trails.csv 

This is the data file containing worm tracks. The data can be processed by the user with his own algorithm to determine additional outputs (such as particle vector direction, distribution of particles within plate, etc). Data is structured in the following way: 

*Additional information includes worm shape 

rundata.dat 

It contains acquisition configuration ( system_ID, folder, project date, acquisition lapse) 

training.dat 

It contains the parameters used by machine learning algorithms 

descriptor.dat 

It contains a sample of box frames used for training 

xy_worms.txt and xy_worms.bin 

It contains information (ascii or binary) about each particle detected on each frame. This data is used by the software to build Worm trails. FAQ 

What are the computer requirements for using the WMicrotracker SMART? 

To use the WMicrotracker SMART, you will need an IBM PC compatible computer with the 

following minimum requirements: 

● Pentium Core i3 processor or above 

● 2 GB of RAM memory 

● 1 available USB port to power the WiFi  access point 

● MS Windows 7 (or higher) operating system 

● At least 1 GB of free hard drive space. 

How far away should the device be from the usb WiFi adapter? 

We recommend placing the WMicrotracker SMART device within 2 meters of the USB Wi-Fi adapter. The adapter can be plugged into any available USB port. You can run the WMicrotracker SMART software from any computer near the device by connecting to the Wi-Fi network "phylumtech.com." 

What brands of plates can I use with the WMicrotracker SMART? 

The WMicrotracker SMART was designed to fit 35mm plates from Greiner Bio-One (GBO). Plates from other brands may not fit in the proper plate adapters. 

Is the WMicrotracker SMART  compatible with all worm st rains? 

The system is compatible with most worm strains, including wild nematode isolates and parasitic nematodes. Non-good swimming strains, such as unc mutants, can also be detected. 

Can I measure movement in C. elegans larvae? 

Yes, you can measure movement in populations of larvae as early as L3. 

How many worms should I put in each plate to get accurate measurements? 

It is recommended to place no more than 20/30 worms per plate. If the worms are too close, the software will not be able to identify each individual worm accurately. 

Is it possible to measure individual worms? 

Yes, it is possible to measure individual worms. We recommend using another plate with at least 5-10 worms to pre-train the software before conducting the experiment. 

How long can I keep and measure my worms in the WMicrotracker SMART? 

Currently, the software allows you to measure only at intervals of 5 minutes. For experiments spanning a longer time period, multiple measurements of 5 minutes can be programmed to be taken at different times. Can I control/change the temperature of my samples with the  WMicrotracker 

SMART ?

The temperature in the WMicrotracker SMART cannot be preset and will depend on the room temperature. However, it is very compact and can easily fit in your incubator if necessary. 

How do I know which plate adapter to use for my experiments? 

If you're detecting worms cultured on NGM, use the petri dish with the lid downwards on the appropriate plate adapter ("lid down"). For small aquatic animals and insects larger than 1mm, use the petri dish with the lid upwards on the appropriate plate adapter ("lid up"). If you're working with organisms smaller than 1mm, use the adapter with the grid (IR Microbeam Light Scattering MODE). 

Can I use NGM with a lawn of bacteria? 

Yes, you can use NGM with a bacteria lawn. However, verify that your bacteria lawn is not too 

dense, as this can prevent recognition of the worm’s silhouette. 

Is it important to manually adjust the lens focus of the WMicrotracker SMART? Is it necessary to change the focus between plates? 

It is essential to adjust the lens focus for your worm layer during the initial setup. If you use the same volume of media on different plates, it will not be necessary to change the focus between plates of the same experimental set. 

I can’t set the focus for the whole area of the plate, what should I pay attention to? 

When preparing NGM plates, avoid moving them, as they can create a natural curvature (meniscus), making it difficult to focus the whole area of the plate. 

What should I use to train the software for the recognition of the silhouette of the worms? 

Use a plate with at least 5-10 worms, selecting worms in different areas of the plates. Avoid selecting worms that are on the edge of the plate. 

Could I re-entrain after the acquisition? 

Yes, you can re-entrain the software after the acquisition. 

Should I stimulate the plate before measurement? 

For reproducible results it is recommended to stimulate each plate just before each measurement. You could stimulate mechanically (tap-tap) or using blue light. 

Where should I place the red line of the circle? 

The red circle line serves as a relative caliper for distance, telling the software how much that diameter is in reality. For instance, if the red circle line is on the edge of the 35mm baseplate, you need to input into the software that the actual measurement is 35mm. Photograph of a plate with the adapter lid down –

Location of the red line. 

How can I avoid worms crawling out of the agar? 

For short experiments, you can use Cu as a repellent. You can either use a ring of copper or pipet 75ul of 100mM CuSO4 solution in the edge of the plate before placing the worms. 

Can I change the measured intervals to be reported? 

The WMicrotracker SMART is constantly detecting and recording movement. When you export your 

data to an excel sheet, you can change the “Bin Size” drop down menu to the desired time interval. The exported data sheet will then show you the average activity count at the intervals you 

specified. 

I see some spots in the detection area, how can I clean them? 

Particles and dust can enter the detection area. Use a microfiber cloth to clean the tray compartment. 

Why do I observe in the report a list of particles (particle trials ID) higher than the number of worms I put? 

When the software stops detecting a worm's silhouette for 10 seconds, that particle's path ends, and the software will assign a new particle ID to that worm when it detects it again. For more information contact us 

info@phylumtech.com
