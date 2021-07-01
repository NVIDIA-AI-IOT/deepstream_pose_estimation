// Copyright 2020 - NVIDIA Corporation
// SPDX-License-Identifier: MIT

#include "post_process.cpp"

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>

#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

#define EPS 1e-6

#define MAX_DISPLAY_LEN 64

// Como yo se que lo voy a correr en Jetson (TEGRA), fuerzo esta definicion para
// que vscode me resalte las partes correctas
#define PLATFORM_TEGRA

// Reemplazo un if 0 de preprocesador por este para poder habilitar y deshabilitar
#define ENABLE_DISPLAY

// CAMERA o FILEIN son excluyentes. Hay que seleccionar una entrada o la otra.
// Versión compilada para entrada desde cámara
//#define CAMERA
#undef CAMERA

// Version compilada para entrada FILE
#undef FILEIN
//#define FILEIN

// Versión compilada para entrada SHMSRC
#define SHMSRC


// Salida a archivo mp4. Puede habilitarse o deshabilitarse independientemente de las otras opciones.
// Me falta hacer uno igual para la salida de video. Warning, se puede habilitar FILEOUT pero el archivo no se ve.
//#undef FILEOUT
#define FILEOUT

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */

/* Le pongo 720 porque es lo que usamos en el prototipo futbol
* pero no es necesariamente la mejor resolución posible
* hay que encontrar cual es la ideal en el trade off -calidad de detección/performance-
*/

#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;

// Un intento muy berreta (variable global) de setear el color del label en funcion de la pose
float rojo = 0.0;

/*Method to parse information returned from the model*/
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec3D<int> peaks;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 2;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  /* Finding peaks within a given window */
  find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
  /* Non-Maximum Suppression */
  Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
  /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
  Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
  /* Assign weights to all edges in the bipartite graph generated */
  Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
  /* Connecting all the Body Parts and Forming a Human Skeleton */
  Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
  return {objects, refined_peaks};
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int count = objects.size();
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  // Draw circles at peaks
  for (auto &object : objects)
  {
    int C = object.size();
    for (int j = 0; j < C; j++)
    {
      int k = object[j];
      if (k >= 0)
      {
        auto &peak = normalized_peaks[j][k];
        int x = peak[1] * MUXER_OUTPUT_WIDTH;
        int y = peak[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        // Acá parece definirse como se dibuja el circulo en cada keypoint
        // Es decir que en cparams.xc y cparams.yx están las coordenadas.
        // En este punto no están asociadas a ninguna persona en particular.
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;

        //g_print("xc: %u\n",cparams.xc);

        cparams.yc = y;
        cparams.radius = 7;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }

    // Draw lines between peaks
    for (int k = 0; k < K; k++)
    {
      int c_a = topology[k][2];
      int c_b = topology[k][3];
      if (object[c_a] >= 0 && object[c_b] >= 0)
      {
        auto &peak0 = normalized_peaks[c_a][object[c_a]];
        auto &peak1 = normalized_peaks[c_b][object[c_b]];
        int x0 = peak0[1] * MUXER_OUTPUT_WIDTH;
        int y0 = peak0[0] * MUXER_OUTPUT_HEIGHT;
        int x1 = peak1[1] * MUXER_OUTPUT_WIDTH;
        int y1 = peak1[0] * MUXER_OUTPUT_HEIGHT;
        if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        // Acá parece definir las líneas entre keypoints
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 2;
        lparams.line_color = NvOSD_ColorParams{0, 0, 255, 1};
        dmeta->num_lines++;
      }
    }
  }
}

/* pgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
// Este es el primer lugar donde tengo todos los resultados de la inferencia.
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  // Iterate over frames
  // Entiendo que esto es porque pueden venir en batches de mas de 1 frame.
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_user = frame_meta->frame_user_meta_list; l_user != NULL;
         l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
      {
        NvDsInferTensorMeta *tensor_meta =
            (NvDsInferTensorMeta *)user_meta->user_meta_data;
        Vec2D<int> objects;
        Vec3D<float> normalized_peaks;
        tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);

        // Esto funciona, imprime guión "-" si no se detecta ese elemento o asterisco "*" si se detecta.
        // Quizás por la red que estoy usando, hay algunos falsos positivos, que claramente no tienen que ver con una persona
        // Tengo que definir un umbral de cantidad de partes detectadas para considerar detección válida. 
        // En la práctica lo que me importa (ver mas abajo) varía según lo que quiero detectar. 
        // Para brazos arriba elijo tener 3 o mas puntos de la cara y las 2 manos.
        g_print("Personas: %lu \n", objects.size()); 
        for (int i = 0; i < objects.size(); i++)
        {
          g_print(" Persona %u: ",i);
          int score=0;
          for (int j = 0; j < objects[i].size(); j++)
          {
              score=score+objects[i][j];
              //g_print("el %u: %d ", j,objects[i][j]);
              //g_print("%d ", objects[i][j]);
              if (objects[i][j] >=0) 
              {
                g_print("* ");
              } else {
                g_print("- ");
              }
          }
          g_print("Score %d \n",score);
        
          // Tengo el score, pero puede ser un caso particular donde no hay cara ni manos.
          // Obviamente en una version mas performante de esto, sólo para detectar brazos arriba, no me importa todo el bloque anterior. Sólo tengo que mirar 0-4, 9 y 10.

          // Chequeo si hay cara (por lo menos 3 puntos del 0 al 4)
          int cara = objects[i][0]+objects[i][1]+objects[i][2]+objects[i][3]+objects[i][4];
          //g_print("cara %d \n",cara);

          // Primero lo pongo acá para probar, luego tiene que estar dentro del loop y finalmente sólo cuando detecto algo útil
          // Hay que hacer query al pad porque pipeline no es global.
          // Hay que ysar ...pad_peer... porque sin el peer no funciona, devuelve siempre 0.
          gint64 current_position;
          gboolean ret;
          //ret = gst_pad_peer_query_position (pad, GST_FORMAT_TIME, &current_position);
          //if (ret) {
          //  g_print("Position %" GST_TIME_FORMAT " \n", GST_TIME_ARGS(current_position));
          //} else {printf("Error");}

          // Chequeo si esta persona tiene cara y 2 manos visibles
          int cantidad_elementos_cara=0;
          float altura_total_cara=0;
          float altura_promedio_cara=0;
          float altura_promedio_manos=0;
          if (cara >= -2 && objects[i][9]>=0 && objects[i][10]>=0)
          {
            g_print(" Tengo objetos suficientes para comparar y detectar brazos arriba\n");
            // Acá va alguna lógica para determinar alto promedio de elementos cara y alto promedio elementos manos y luego comparar

            // Itero sobre los objetos cara (0 al 4), si está, sumo su altura y lo cuento
            for (int j = 0; j < 5;j++)
            {
              if (objects[i][j]>=0)
              {
                cantidad_elementos_cara++;
                // normalized_peaks[objeto][persona][coordenada, (x y )0 / 1 ]
                altura_total_cara=altura_total_cara+normalized_peaks[j][i][0];
              }
            }
            g_print(" Hay %d elementos en la cara.\n",cantidad_elementos_cara);
            // Terminé de iterar sobre los 5 objetos cara, calculo el promedio
            altura_promedio_cara=altura_total_cara/cantidad_elementos_cara;
            g_print(" Altura promedio elementos cara %.3f\n",altura_promedio_cara);

            // Si estoy acá dentro, existen los dos elementos mano. Calculo altura promedio manos.
            altura_promedio_manos=(normalized_peaks[9][i][0]+normalized_peaks[10][i][0])/2;
            g_print(" Altura promedio elementos mano %.3f\n",altura_promedio_manos);
          }
     
          // A esto lo tengo que revisar, porque si una persona está ON y otras OFF, me queda del color
          // de la última persona. Tendria que hacer un OR entre las distintas personas de este cuadro.
          if (altura_promedio_manos<altura_promedio_cara)
          {
             g_print(" BRAZOS ARRIBA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
             rojo = 1.0;
             // Query stream position to mark the highlight
             ret = gst_pad_peer_query_position (pad, GST_FORMAT_TIME, &current_position);
              if (ret) {
                g_print("Position %" GST_TIME_FORMAT " \n", GST_TIME_ARGS(current_position));
              } else {printf("Error");}

              
          } else { rojo = 0.0; }

          // Trato de buscar las coordenadas de un punto (0).
/*
          // i es persona, 0 es nariz
          auto &peak = normalized_peaks[0][i];
          int x = peak[1] * MUXER_OUTPUT_WIDTH;
          int y = peak[0] * MUXER_OUTPUT_HEIGHT;
          g_print("Coordenadas nariz: %d,%d\n",x,y);

                    // i es persona, 1 es ojo left
          auto &peak2 = normalized_peaks[1][i];
          x = peak2[1] * MUXER_OUTPUT_WIDTH;
          y = peak2[0] * MUXER_OUTPUT_HEIGHT;
          g_print("Coordenadas ojo izquierdo: %d,%d\n",x,y);

*/
          // normalized_peaks[keypint][persona][x/y]
          // Acá tengo que ver algunos segfaults, creo que antes de llamar a normalized_peaks hay que verificar que el objeto esté en 'objects'.
          //g_print("Coordenadas keypoint 0 (nose): %f,%f, 9 (left hand): %f,%f, 10 (right hand): %f,%f\n",
          //                           normalized_peaks[0][i][0],normalized_peaks[0][i][1],
          //                           normalized_peaks[9][i][0],normalized_peaks[9][i][1],
          //                           normalized_peaks[10][i][0],normalized_peaks[10][i][1]);

        }

        create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
      }
    }

    // Dentro de cada frame, itero sobre los objetos. 
    // Esto no parece estar corriendo, posiblemente porque no hago BATCH.
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
           l_user = l_user->next)
      {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
        {
          NvDsInferTensorMeta *tensor_meta =
              (NvDsInferTensorMeta *)user_meta->user_meta_data;
          Vec2D<int> objects;
          Vec3D<float> normalized_peaks;
          tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);

          // Intento mostrar contenido de los vectores
          //g_print("objects size %lu \n", 10); 
          g_print(".");

          create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "TECNOX LABS - Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = "Mono";
    txt_params->font_params.font_size = 16;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    //txt_params->text_bg_clr.red = 1.0;
    txt_params->text_bg_clr.red = rojo;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 0.5;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of Stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }

  default:
    break;
  }
  return TRUE;
}

gboolean
link_element_to_tee_src_pad(GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(tee), "src_%u");
  tee_src_pad = gst_element_request_pad(tee, padtemplate, NULL, NULL);

  if (!tee_src_pad)
  {
    g_printerr("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad(sinkelem, "sink");
  if (!sinkpad)
  {
    g_printerr("Failed to get sink pad from '%s'",
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }

  if (gst_pad_link(tee_src_pad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link '%s' and '%s'", GST_ELEMENT_NAME(tee),
               GST_ELEMENT_NAME(sinkelem));
    goto done;
  }
  ret = TRUE;

done:
  if (tee_src_pad)
  {
    gst_object_unref(tee_src_pad);
  }
  if (sinkpad)
  {
    gst_object_unref(sinkpad);
  }
  return ret;
}

int main(int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstCaps *caps = NULL, *caps2 = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
             *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL,
             *nvvideoconvert = NULL, *tee = NULL, *h264encoder = NULL, *cap_filter = NULL, *filesink = NULL, *queue = NULL, *qtmux = NULL, *h264parser1 = NULL, *nvsink = NULL;

  // Creo algunos elementos mas para la entrada de cámara
  GstElement *camera = NULL, *camera_caps=NULL, *camera_caps2=NULL, *camera_conv=NULL;

// Check for wrong settings FILEIN & CAMERA

#ifdef FILEIN
#ifdef CAMERA
  g_print("ERROR. Compilado con entrada FILEIN y CAMERA al mismo tiempo. Se debe elegir uno solo.\n");
  return -1;
#endif
#endif


/* Add a transform element for Jetson*/
// If DISPLAY & TEGRA
#ifdef ENABLE_DISPLAY
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
#endif

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

#ifdef FILEIN
  g_print("Compilado con entrada FILEIN (no tiene que estar habilitado CAMERA ni SHMSRC)\n");
  /* Check input arguments */
  if (argc != 3)
  {
    g_printerr("Compiled with define FILEIN\nUsage: %s <filename> <output-path>\n", argv[0]);
    return -1;
  }
#endif

#ifdef SHMSRC
  g_print("Compilado con entrada SHMSRC (no tiene que estar habilitado CAMERA ni FILEIN)\n");
#endif

#ifdef ENABLE_DISPLAY
// This define is not yet programmed. It won't work.
  g_print("Compilado con ENABLE_DISPLAY\nATENCIÓN SETEAR env DISPLAY!!!\n");
#endif

#ifdef FILEOUT
  g_print("Compilado con salida a archivo FILEOUT\n");
#endif

#ifdef CAMERA
g_print("Compilado con entrada CAMERA (no tiene que estar habilitado FILEIN)\n");
#ifdef FILEOUT
  /* Check input arguments */
  if (argc != 2)
  {
    g_printerr("Compiled with define CAMERA\nUsage: %s <output-path>\n", argv[0]);
    return -1;
  }
#endif
#endif


  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

#ifdef FILEOUT
  h264parser1 = gst_element_factory_make("h264parse", "h264-parser1");
#endif

  #ifdef FILEIN
    /* Source element for reading from the file */
    source = gst_element_factory_make("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
    * we need a h264parser */
    h264parser = gst_element_factory_make("h264parse", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");
  #endif

#ifdef SHMSRC
source = gst_element_factory_make("shmsrc", "sharedmemory-source");
if (!source)
  {
    g_printerr("SHMSRC Source element could not be created. Exiting.\n");
    return -1;
  }
shm_caps = gst_element_factory_make("capsfilter", "shmsrc_caps");
  caps = gst_caps_from_string("video/x-raw,width=1280,height=720,format=I420,framerate=(fraction)5/1,pixel-aspect-ratio=(fraction)1/1,interlace-mode=(string)progressive");
  g_object_set(G_OBJECT(shm_caps), "caps", caps, NULL);

#endif

  #ifdef CAMERA
  //camera = gst_element_factory_make("nvv4l2camerasrc","nvv4l2camerasrc");
  // Prové nvv4l2camerasrc en un momento en el que con v4l2src no arrancaba el pipeline.
  // Es probble que nvv se mejor, porque directamente pone los datos en NVMM, pero nosotros en definitiva vamos a usar shmsrc
  camera = gst_element_factory_make("v4l2src","v4l2src");
  if (!camera)
  {
    g_printerr("Camera element could not be created. Exiting.\n");
    return -1;
  }
  camera_caps = gst_element_factory_make("capsfilter", "camera_caps");
  // nvv4l2... caps = gst_caps_from_string("video/x-raw(memory:NVMM), width=1280, height=720,framerate=10/1,interlace-mode=(string)progressive");
  caps = gst_caps_from_string("video/x-raw, width=1280, height=720,framerate=10/1");
  g_object_set(G_OBJECT(camera_caps), "caps", caps, NULL);

  camera_conv = gst_element_factory_make("nvvideoconvert","cameraconv");
  if (!camera_conv)
  {
    g_printerr("Camera convert element could not be created. Exiting.\n");
    return -1;
  }
  camera_caps2 = gst_element_factory_make("capsfilter", "camera_caps2");
  caps2 = gst_caps_from_string("video/x-raw(memory:NVMM), width=1280, height=720,framerate=10/1, format=NV12");
  g_object_set(G_OBJECT(camera_caps2), "caps", caps2, NULL);
  #endif

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element pipeline/streammux could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

#ifdef FILEOUT
  queue = gst_element_factory_make("queue", "queue");
  filesink = gst_element_factory_make("filesink", "filesink");
#endif

  /* Set output file location */
  #ifdef FILEIN
  char *output_path = argv[2];
  #endif

  #ifdef CAMERA
  #ifdef FILEOUT
  char *output_path = argv[1];
  #endif
  #endif

#ifdef FILEOUT
  strcat(output_path,"Pose_Estimation.mpeg");
//  strcat(output_path,"Pose_Estimation.h264");
  g_object_set(G_OBJECT(filesink), "location", output_path, "async", FALSE, NULL); // ojo agregue lo de async. Sin esto y con entrada cámara, se cuelga en el frame 0.
#endif
  
  nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
  tee = gst_element_factory_make("tee", "TEE");

#ifdef FILEOUT
  h264encoder = gst_element_factory_make("nvv4l2h264enc", "video-encoder");
  cap_filter = gst_element_factory_make("capsfilter", "enc_caps_filter");
  caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
  g_object_set(G_OBJECT(cap_filter), "caps", caps, NULL);
  //qtmux = gst_element_factory_make("qtmux", "muxer");
  qtmux = gst_element_factory_make("mpegtsmux", "muxer");  // Es la que me funcionó. Algo no anda con mp4.
#endif

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef ENABLE_DISPLAY
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
#endif

  nvsink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
  sink = gst_element_factory_make("fpsdisplaysink", "fps-display");

  g_object_set(G_OBJECT(sink), "text-overlay", FALSE, "video-sink", nvsink, "sync", FALSE, NULL);
#endif // ENABLE_DISPLAY

#ifdef FILEIN
  if (!source || !h264parser || !decoder || !pgie || !nvvidconv || !nvosd || !sink || !cap_filter || !tee || !nvvideoconvert ||
      !h264encoder || !filesink || !queue || !qtmux || !h264parser1)
  {
    g_printerr("One element FILE could not be created. Exiting.\n");
    return -1;
  }
#endif

#ifdef ENABLE_DISPLAY
if (!sink)
{ g_printerr("Sink element could not be created. Exiting.\n"); return -1;}
#endif

#ifdef SHMSRC
#ifdef FILEOUT
  if (!source || !source_caps || !pgie || !nvvidconv || !nvosd || !cap_filter || !tee || !nvvideoconvert ||
      !h264encoder || !filesink || !queue || !qtmux || !h264parser1)
  {
    g_printerr("One element SHMSRC -with fileout enabled- could not be created. Exiting.\n");
    return -1;
  }
#else
// esta rama no está probada
if (!source || !source_caps || !pgie || !nvvidconv || !nvosd || !tee )
  {
    g_printerr("One element SHMSRC -no fileout enabled- could not be created. Exiting.\n");
    return -1;
  }
#endif
#endif

#ifdef CAMERA
#ifdef FILEOUT
  if (!camera || !camera_caps || !camera_caps2 || !camera_conv || !pgie || !nvvidconv || !nvosd || !cap_filter || !tee || !nvvideoconvert ||
      !h264encoder || !filesink || !queue || !qtmux || !h264parser1)
  {
    g_printerr("One element CAMERA -with fileout enabled- could not be created. Exiting.\n");
    return -1;
  }
#else
if (!camera || !camera_caps || !camera_caps2 || !camera_conv || !pgie || !nvvidconv || !nvosd || !tee )
  {
    g_printerr("One element CAMERA -no fileout enabled- could not be created. Exiting.\n");
    return -1;
  }
#endif
#endif

#ifdef ENABLE_DISPLAY
#ifdef PLATFORM_TEGRA
  if (!transform)
  {
    g_printerr("Tegra EGLTRANSFORM element could not be created. Exiting.\n");
    return -1;
  }
#endif
#endif // ENABLE DISPLAY

#ifdef FILEIN
  /* we set the input filename to the source element */
  g_object_set(G_OBJECT(source), "location", argv[1], NULL);
#endif

#ifdef SHMSRC
  /* we set the socket path to the shared memory source element */
  g_object_set(G_OBJECT(source), "socket-path", "/tmp/shmsink", NULL);
#endif


#ifdef CAMERA
/* we set the input camera to the source element */
  // nvv... g_object_set(G_OBJECT(camera), "device", "/dev/video0", "bufapi-version", TRUE, NULL);
  g_object_set(G_OBJECT(camera), "device", "/dev/video0", NULL);
  // no funciona sin el bufapi g_object_set(G_OBJECT(camera), "device", "/dev/video0", NULL);
#endif

#ifdef FILEIN
  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 1,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
#endif

#ifdef CAMERA
  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 1,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, 
               "live-source", TRUE, NULL);
#endif

#ifdef SHMSRC
  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 1,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, 
               "live-source", TRUE, NULL);
#endif

#ifdef FILEOUT
g_object_set(G_OBJECT(h264parser1), "config-interval", 1, NULL);
#endif

  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set(G_OBJECT(pgie), "output-tensor-meta", TRUE,
               "config-file-path", "deepstream_pose_estimation_config.txt", NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */

// FILE
#ifdef FILEIN
  gst_bin_add_many(GST_BIN(pipeline),
                   source, h264parser, decoder, streammux, pgie,
                   nvvidconv, nvosd, transform, sink,
                   tee, nvvideoconvert, h264encoder, cap_filter, filesink, queue, h264parser1, qtmux, NULL);
#endif

// CAM
#ifdef CAMERA
#ifdef FILEOUT
  gst_bin_add_many(GST_BIN(pipeline),
                   camera, camera_conv, camera_caps, camera_caps2, streammux, pgie,
                   nvvidconv, nvosd, tee, queue, nvvideoconvert, 
                   h264encoder, cap_filter, h264parser1, qtmux, filesink, NULL);
#else
gst_bin_add_many(GST_BIN(pipeline),
                   camera, camera_conv, camera_caps, camera_caps2, streammux, pgie,
                   nvvidconv, nvosd, tee, NULL);
#endif
#endif

// SHARED MEMORY SOURCE
#ifdef SHMSRC
#ifdef FILEOUT
// aca reuso elementos camera_conv y camera_caps2 a pesar de que es para shmsrc
// camera_conv mete el pipeline en NVMM
  gst_bin_add_many(GST_BIN(pipeline),
                   source, camera_conv, source_caps, camera_caps2, streammux, pgie,
                   nvvidconv, nvosd, tee, queue, nvvideoconvert, 
                   h264encoder, cap_filter, h264parser1, qtmux, filesink, NULL);
#else
gst_bin_add_many(GST_BIN(pipeline),
                   source, camera_conv, source_caps, camera_caps2, streammux, pgie,
                   nvvidconv, nvosd, tee, NULL);
#endif
#endif

#ifdef ENABLE_DISPLAY
gst_bin_add_many(GST_BIN(pipeline),transform,sink,NULL);
#endif

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
  if (!sinkpad)
  {
    g_printerr("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

#ifdef FILEIN
  srcpad = gst_element_get_static_pad(decoder, pad_name_src);
  if (!srcpad)
  {
    g_printerr("Decoder request src pad failed. Exiting.\n");
    return -1;
  }
#endif

#ifdef CAMERA
  srcpad = gst_element_get_static_pad(camera_caps2, pad_name_src);
  if (!srcpad)
  {
    g_printerr("camera_caps2 request src pad failed. Exiting.\n");
    return -1;
  }
#endif

#ifdef SHMSRC
  srcpad = gst_element_get_static_pad(camera_caps2, pad_name_src);
  if (!srcpad)
  {
    g_printerr("Compilado con SHMSRC: camera_caps2 request src pad failed. Exiting.\n");
    return -1;
  }
#endif

  if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link decoder/camera_caps2 to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref(sinkpad);
  gst_object_unref(srcpad);


#ifdef FILEIN
  if (!gst_element_link_many(source, h264parser, decoder, NULL))
  {
    g_printerr("Source Elements FILE could not be linked. Exiting.\n");
    return -1;
  }
#endif

#ifdef CAMERA
  if (!gst_element_link_many(camera,camera_caps, camera_conv, camera_caps2, NULL))
  {
    g_printerr("Source Elements CAMERA could not be linked. Exiting.\n");
    return -1;
  }
#endif

#ifdef SHMSRC
  if (!gst_element_link_many(source,source_caps, camera_conv, camera_caps2, NULL))
  {
    g_printerr("Source Elements SHMSRC could not be linked. Exiting.\n");
    return -1;
  }
#endif


#ifdef FILEOUT

if (! gst_element_link(streammux,pgie))
  { g_printerr("Cannot link streammux with pgie. Exiting.\n"); return -1; }

if (! gst_element_link(pgie,nvvidconv))
  { g_printerr("Cannot link pgie with nvvidconv. Exiting.\n"); return -1; }

if (! gst_element_link(nvvidconv, nvosd))
  { g_printerr("Cannot link nvvidconv with nvosd. Exiting.\n"); return -1; }

if (! gst_element_link(nvosd, tee))
  { g_printerr("Cannot link nvosd with tee. Exiting.\n"); return -1; }

#else
  if (!gst_element_link_many(streammux, pgie,
                             nvvidconv, nvosd, tee, NULL))
  { g_printerr("Elements could not be linked: 2. Exiting.\n"); return -1; }
  
#endif

// Comun a FILEOUT o no FILEOUT
#ifdef ENABLE_DISPLAY
if (! gst_element_link(tee, transform))
  { g_printerr("Cannot link tee with transform. Exiting.\n"); return -1; }

  if (! gst_element_link(transform, sink))
  { g_printerr("Cannot link transform with sink. Exiting.\n"); return -1; }

#endif


#if 0
  if (!link_element_to_tee_src_pad(tee, queue)) {
      g_printerr ("Could not link tee to sink\n");
      return -1;
  }
  if (!gst_element_link_many (queue, transform, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif

#ifdef FILEOUT
  if (!link_element_to_tee_src_pad(tee, queue))
  {
    g_printerr("Could not link tee to nvvideoconvert\n");
    return -1;
  }
  //if (!gst_element_link_many(queue, nvvideoconvert, cap_filter, h264encoder,
  //                           h264parser1, qtmux, filesink, NULL))
  if (!gst_element_link_many(queue, nvvideoconvert, cap_filter, h264encoder, h264parser1, qtmux, filesink, NULL))
  
  {
    g_printerr("Elements FILEOUT could not be linked\n");
    return -1;
  }
#endif

  GstPad *pgie_src_pad = gst_element_get_static_pad(pgie, "src");
  if (!pgie_src_pad)
    g_print("Unable to get pgie src pad\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      pgie_src_pad_buffer_probe, (gpointer)sink, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, (gpointer)sink, NULL);

  /* Set the pipeline to "playing" state */
  #ifdef FILEIN
  g_print("Now playing: %s\n", argv[1]);
  #endif

  #ifdef CAMERA
  g_print("Now playing camera input\n");
  #endif

#ifdef SHMSRC
  g_print("Now playing shared memory input\n");
#endif

#ifdef ENABLE_DISPLAY
g_print("Compiled with ENABLE_DISPLAY\n");
#else
g_print("Compiled without ENABLE_DISPLAY\n");
#endif

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // DEBUG DOT GRAPH
  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
