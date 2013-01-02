//
//  main.cpp
//  CroudsourceID
//
//  Created by Saqoosha on 12/11/16.
//  Copyright (c) 2012 Saqoosha. All rights reserved.
//

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <Poco/Glob.h>
#include <math.h>
#include <pthread.h>
#include "picojson.h"


using namespace std;
using namespace cv;


#define MAX_NUM_RESULTS 5
#define NUM_THREADS 8


vector<string> icon_files;
Mat features;
flann::Index icon_index;
float match_threshold = 1.f;
typedef map<string, Mat> ImageCache;
ImageCache icon_cache;


// 16x16px の画像を読み込んで flann の index 用に整形
void loadImage(string filename) {
  Mat image = imread(filename);
  Mat data8u = image.reshape(1, 1); // 1 次元 1 チャネルに
  if (features.rows == icon_files.size()) { // いっぱいなら
    features.resize(features.rows * 2); // とりあえず倍に
    cout << "num icons: " << features.rows << endl;
  }
  Mat target(features, Rect(0, (int)icon_files.size(), 768, 1));
  Mat data32f;
  data8u.convertTo(data32f, CV_32FC1); // 32bit float に
  data32f.copyTo(target);
  
  filename.replace(filename.find("/16/"), 4, "/256/");
  icon_files.push_back(filename);
}


// flann の index つくる
void buildIndex() {
  icon_files.clear();
  features.create(1024, 768, CV_32FC1); // 768 次元。とりあえず 1024 画像ぶん。

  double start = getTickCount();
  
  set<string> files;
  // ディレクトリの中の png ぜんぶリストアップ
  Poco::Glob::glob("/Users/hiko/Dropbox/Dev/openFrameworks/of_v0072_osx_release/apps/myApps/CrowdsourceID/bin/data/resized/16/*/*/*.png", files);
  set<string>::iterator it = files.begin();
  for (; it != files.end(); ++it) {
    loadImage(*it);
  }
  cout << "load images: " << (getTickCount() - start) / getTickFrequency() << " secs" << endl;

  features.resize(icon_files.size()); // 最終的な画像数に
  cout << "features size: " << features.rows << endl;

  start = getTickCount();
  icon_index.build(features, flann::KDTreeIndexParams()); // index つくる
  cout << "build index: " << (getTickCount() - start) / getTickFrequency() << " secs" << endl;
}


// roi で指定された領域を filename の画像におきかえる
void replaceImage(Mat &roi, string filename) {
  Mat icon, resized;
  ImageCache::iterator it = icon_cache.find(filename);
  if (it == icon_cache.end()) { // まだ読み込んでないファイルなら
    icon = imread(filename);
    icon_cache[filename] = icon;
  } else {
    icon = it->second;
  }
  
  if (icon.data != NULL) { // 読み込みエラーじゃなければ
    resize(icon, resized, roi.size(), 0, 0, INTER_AREA); // 指定エリアのサイズにリサイズして
    resized.copyTo(roi); // コピー
  } else {
    roi = Scalar(255, 0, 0);
    cout << icon.cols << ", " << icon.rows << ": " << filename << endl;
  }
}


// target エリアをモザイクにして result に出力。min_size まで分割しておｋ
bool placeIcon(Mat &target, Mat &result, int min_size) {
  static int count = 0;
  Mat roi32, query;
  vector<int> indices(MAX_NUM_RESULTS);
  vector<float> dists(MAX_NUM_RESULTS);
  target.convertTo(roi32, CV_32F); // flann で検索できる形式に
  resize(roi32, query, Size(16, 16), 0, 0, INTER_AREA); // 変換
  query = query.reshape(1, 1); // する
  icon_index.knnSearch(query, indices, dists, MAX_NUM_RESULTS); // んで検索
  int n = 0;
  while (n < MAX_NUM_RESULTS && sqrtf(dists[n]) / 768.f < match_threshold) { // match_threshold 以下のスコアの結果はなんこ？
    ++n;
  }
  if (target.cols <= min_size || n > 0) { // min_size 以下、か、マッチした画像があったら
    replaceImage(result, icon_files[indices[count++ % MAX(1, n)]]); // アイコンはめこむ
  } else { // だめだったら縦横半分に分割してそれぞれまた調べる
    int left, right, top, bottom;
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        left = x * target.cols * .5f;
        right = (x + 1) * target.cols * .5f;
        top = y * target.rows * .5f;
        bottom = (y + 1) * target.rows * .5f;
        Mat target_roi(target, Rect(left, top, right - left, bottom - top));
        left = x * result.cols * .5f;
        right = (x + 1) * result.cols * .5f;
        top = y * result.rows * .5f;
        bottom = (y + 1) * result.rows * .5f;
        Mat result_roi(result, Rect(left, top, right - left, bottom - top));
        placeIcon(target_roi, result_roi, min_size);
      }
    }
  }
  return true;
}


// original 画像をモザイクにして result に出力。min_size から max_size のあいだでぶんかつ。
void buildMosaic(Mat &original, Mat &result, int scale, int max_size, int min_size) {
  for (int y = 0; y < ceil((float)original.rows / max_size); ++y) {
    for (int x = 0; x < ceil((float)original.cols / max_size); ++x) {
      Mat target_roi(original, Rect(x * max_size, y * max_size, max_size, max_size));
      Mat result_roi(result, Rect((x * max_size) * scale, (y * max_size) * scale, max_size * scale, max_size * scale));
      placeIcon(target_roi, result_roi, min_size);
    }
  }
}


// filename の画像ファイルをよみこんで mosaic_file に書きだす。
void buildMosaic(string filename, string mosaic_file) {
  double start = getTickCount();

  Mat original = imread(filename, CV_LOAD_IMAGE_COLOR);
  
  int scale = 1;
  Mat result(original.rows * scale, original.cols * scale, CV_8UC3, Scalar(0xff0000));
  buildMosaic(original, result, scale, 128, 8);
  
  imwrite(mosaic_file, result);

  cout << "build mosaic: " << (getTickCount() - start) / getTickFrequency() << " secs" << endl;
  
  imshow("mosaiced", result);
  waitKey();
}


VideoCapture vin;
int video_width;
int video_height;
picojson::array *root;
pthread_mutex_t mutex;
int completed = 0;
int total_frames;
int next_frame = 0;


// 各スレッドで実行されるやつ、メインのん
void *threadMain(void *argument) {
  int frame;
  while (next_frame < total_frames) { // 全フレーも処理おわった？
    pthread_mutex_lock(&mutex); // ムービーファイルから画像読み込むとこはマルチスレッドだめ
    frame = next_frame++;
    double start = getTickCount();
    vin.set(CV_CAP_PROP_POS_FRAMES, frame);
    Mat original;
    vin >> original;
    pthread_mutex_unlock(&mutex);
    
    if (original.data == NULL) continue;
    
    picojson::object &info = (*root)[frame].get<picojson::object>(); // 読み込んだフレームのパラメータよみこむ
    
    Mat affine = getRotationMatrix2D(Point2f(0.f, 0.f), info["rotation"].get<double>(), 1); // 元画像をてきとうに変形
    affine.at<double>(0, 2) += info["x"].get<double>();
    affine.at<double>(1, 2) += info["y"].get<double>();
    Mat rotated;
    warpAffine(original, rotated, affine, Size(info["width"].get<double>(), info["height"].get<double>()), INTER_LINEAR, BORDER_REPLICATE);
    
    int scale = 2;
    Mat mosaic(info["height"].get<double>() * scale, info["width"].get<double>() * scale, CV_8UC3, Scalar(0xff0000));
    buildMosaic(rotated, mosaic, scale, info["maxSize"].get<double>(), info["minSize"].get<double>()); // モザイクにする
    
    Mat result;
//    affine = getRotationMatrix2D(Point2f(0.f, 0.f), info["rotation"].get<double>(), 1);
//    affine.at<double>(0, 2) += info["x"].get<double>() * scale;
//    affine.at<double>(1, 2) += info["y"].get<double>() * scale;
    warpAffine(mosaic, result, affine, Size(video_width, video_height) * scale, INTER_LANCZOS4 | WARP_INVERSE_MAP); // んでもとにもどす。
    
    stringstream file;
    file << "/Volumes/Data/_temp/sequence/" << setw(5) << setfill('0') << frame << ".png";
    printf("%d: %s (%.2gsecs)\n", completed++, file.str().c_str(), (getTickCount() - start) / getTickFrequency());
    imwrite(file.str(), result); // できたフレームを書きだす。
    
//    imshow("Mosaic", result);
//    waitKey(10);
  }
  return NULL;
}


// プログラムこっから
int main(int argc, const char * argv[]) {
  vin.open("../../crowdsource_source1221-2.mp4"); // 元ムービー
  if (!vin.isOpened()) {
    exit(1);
  }
  
  video_width = vin.get(CV_CAP_PROP_FRAME_WIDTH);
  video_height = vin.get(CV_CAP_PROP_FRAME_HEIGHT);
  total_frames = vin.get(CV_CAP_PROP_FRAME_COUNT);
  float fps = vin.get(CV_CAP_PROP_FPS);
  cout << "width: " << video_width << endl;
  cout << "height: " << video_height << endl;
  cout << "num frames: " << total_frames << endl;
  cout << "fps: " << fps << endl;

  ifstream json_file("data.json"); // フレームごとのモザイクサイズとかのパラメータ
  picojson::value v;
  json_file >> v;
  root = &v.get<picojson::array>();
  total_frames = MIN((int)root->size(), total_frames) - 1;
  
  buildIndex(); // アイコン素材よみこむ
  
  pthread_mutex_init(&mutex, NULL);
  pthread_t threads[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_create(&threads[i], NULL, threadMain, NULL); // スレッドつくって起動
  }
  for (int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL); // 全部終わるまで待つ
  }
  pthread_mutex_destroy(&mutex);
}

