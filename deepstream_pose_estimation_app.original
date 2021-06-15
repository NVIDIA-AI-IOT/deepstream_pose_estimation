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

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

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
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;
        cparams.yc = y;
        cparams.radius = 8;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }

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
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 3;
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_lines++;
      }
    }
  }
}

/* pgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
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
        create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
      }
    }

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
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number =  %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, "");

    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    txt_params->font_params.font_name = "Mono";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

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
  GstCaps *caps = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
             *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL,
             *nvvideoconvert = NULL, *tee = NULL, *h264encoder = NULL, *cap_filter = NULL, *filesink = NULL, *queue = NULL, *qtmux = NULL, *h264parser1 = NULL, *nvsink = NULL;

/* Add a transform element for Jetson*/
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  /* Check input arguments */
  if (argc != 3)
  {
    g_printerr("Usage: %s <filename> <output-path>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream-tensorrt-openpose-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make("h264parse", "h264-parser");
  h264parser1 = gst_element_factory_make("h264parse", "h264-parser1");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  queue = gst_element_factory_make("queue", "queue");
  filesink = gst_element_factory_make("filesink", "filesink");
  
  /* Set output file location */
  char *output_path = argv[2];
  strcat(output_path,"Pose_Estimation.mp4");
  g_object_set(G_OBJECT(filesink), "location", output_path, NULL);
  
  nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvvideo-converter1");
  tee = gst_element_factory_make("tee", "TEE");
  h264encoder = gst_element_factory_make("nvv4l2h264enc", "video-encoder");
  cap_filter = gst_element_factory_make("capsfilter", "enc_caps_filter");
  caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=I420");
  g_object_set(G_OBJECT(cap_filter), "caps", caps, NULL);
  qtmux = gst_element_factory_make("qtmux", "muxer");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
#endif
  nvsink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
  sink = gst_element_factory_make("fpsdisplaysink", "fps-display");

  g_object_set(G_OBJECT(sink), "text-overlay", FALSE, "video-sink", nvsink, "sync", FALSE, NULL);

  if (!source || !h264parser || !decoder || !pgie || !nvvidconv || !nvosd || !sink || !cap_filter || !tee || !nvvideoconvert ||
      !h264encoder || !filesink || !queue || !qtmux || !h264parser1)
  {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }
#ifdef PLATFORM_TEGRA
  if (!transform)
  {
    g_printerr("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  /* we set the input filename to the source element */
  g_object_set(G_OBJECT(source), "location", argv[1], NULL);

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 1,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

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
#ifdef PLATFORM_TEGRA
  gst_bin_add_many(GST_BIN(pipeline),
                   source, h264parser, decoder, streammux, pgie,
                   nvvidconv, nvosd, transform, /*sink,*/
                   tee, nvvideoconvert, h264encoder, cap_filter, filesink, queue, h264parser1, qtmux, NULL);
#else
  gst_bin_add_many(GST_BIN(pipeline),
                   source, h264parser, decoder, streammux, pgie,
                   nvvidconv, nvosd, /*sink,*/
                   tee, nvvideoconvert, h264encoder, cap_filter, filesink, queue, h264parser1, qtmux, NULL);
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

  srcpad = gst_element_get_static_pad(decoder, pad_name_src);
  if (!srcpad)
  {
    g_printerr("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref(sinkpad);
  gst_object_unref(srcpad);

  if (!gst_element_link_many(source, h264parser, decoder, NULL))
  {
    g_printerr("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }
#if 0
#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many (streammux, pgie,
          nvvidconv, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif
#else
#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many(streammux, pgie,
                             nvvidconv, nvosd, tee, NULL))
  {
    g_printerr("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many(streammux, pgie, nvvidconv, nvosd, tee, NULL))
  {
    g_printerr("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif
#if 0
  if (!link_element_to_tee_src_pad(tee, queue)) {
      g_printerr ("Could not link tee to sink\n");
      return -1;
  }
  if (!gst_element_link_many (queue, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!link_element_to_tee_src_pad(tee, queue))
  {
    g_printerr("Could not link tee to nvvideoconvert\n");
    return -1;
  }
  if (!gst_element_link_many(queue, nvvideoconvert, cap_filter, h264encoder,
                             h264parser1, qtmux, filesink, NULL))
  {
    g_printerr("Elements could not be linked\n");
    return -1;
  }
#endif

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
  g_print("Now playing: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

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
