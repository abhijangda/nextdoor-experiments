#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <assert.h>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <tuple>
#include <iterator>
#include <utility>
#include <stdlib.h>

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#define LINE_SIZE 1024*1024

class Vertex
{
private:
  int id;
  std::vector<std::pair<int, float>> edges;

public:
  Vertex (int _id) : id (_id)
  {
  }

  Vertex ():id(-1){}

  void set_id (int _id) {id = _id;}
  int get_id () {return id;}
  void add_edge (int vertexID, float weight) {edges.push_back (std::make_pair(vertexID, weight));}
  void sort_edges () {std::sort (edges.begin(), edges.end ());}
  void update_edges (std::unordered_map <int, int>& prev_to_new_ids) 
  {
    for (size_t i = 0; i < edges.size (); i++) {
      edges[i].first = prev_to_new_ids[edges[i].first];
    }

    sort_edges ();
  }

  void remove_duplicate_edges () 
  {
    std::set<std::pair<int,float>> set_edges = std::set<std::pair<int,float>> (edges.begin(), edges.end ());
    edges = std::vector<std::pair<int,float>> (set_edges.begin (), set_edges.end ());
    //sort_edges ();
  }

  std::vector <std::pair<int, float>>& get_edges () {return edges;}
  void print (std::ostream& os)
  {
    os << id << " " << " ";
    for (auto edge : edges) {
      os << edge.first << " " << edge.second << " ";
    }

    os << std::endl;
  }

  static bool compare_vertex (Vertex& v1, Vertex& v2) 
  {
    return v1.edges.size () < v2.edges.size ();
  }

  float max_weight()
  {
    float w = 0.0f;

    for (auto p : edges) {
      w = std::max(w, p.second);
    }

    return w;
  }
};

int chars_in_int (int num)
{
  if (num == 0) return sizeof(char);
  return (int)((ceil(log10(num))+1)*sizeof(char));
}

class Graph
{
public:
  std::vector<Vertex> vertices;
  std::vector<std::pair<int,int>> edges;
  int n_edges;

public:
  Graph (std::vector<Vertex> _vertices, int _n_edges) :
    vertices (_vertices), n_edges(_n_edges)
  {}
  Graph ()
  {}
  enum {
    EdgeList,
    AdjacenyList,
  } GraphFileType;

  void load_from_edge_list_binary (std::string file_path, bool weighted) 
  {
    FILE* fp = fopen (file_path.c_str(), "rb");
    if (fp == nullptr) {
      std::cout << "File '" << file_path << "' not found" << std::endl;
      return;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *string = new char[fsize + 1];
    if (fread(string, 1, fsize, fp) != (size_t)fsize) {
      std::cout << "" << std::endl;
      abort();
    }
    std::cout << "Graph Binary Loaded" << std::endl;

    n_edges = 0;
    
    for (size_t s = 0; s < (size_t)fsize; s += 12) {
      int src = *(int*)(string+s);
      int dst = *(int*)(string+s+4);
      float weight = *(float*)(string+s+8);

      if ((size_t)src >= vertices.size()) {
        vertices.resize(src+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      if ((size_t)dst >= vertices.size()) {
        vertices.resize(dst+1);
        // for (size_t i = sz; i <= src; i++) {
        //   vertices.push_back(Vertex(i, i));
        // }
      }

      vertices[src].add_edge(dst, weight);
      edges.push_back(std::make_pair(src,dst));

      n_edges++;
    }

    delete string;
    fclose(fp);

    for (auto v : vertices) {
      v.sort_edges();
    }
  }

  const std::vector<Vertex>& get_vertices () {return vertices;}
  int get_n_edges () {return n_edges;}

  void print (std::ostream& os) 
  {
    for (auto v : vertices) {
      v.print (os);
    }
  }
};

Graph graph;
extern "C"
void loadgraph(char* path)
{
  graph.load_from_edge_list_binary(path, true);
}
extern "C"
int numberOfEdges()
{
  return graph.get_n_edges();
}
extern "C"
int numberOfVertices()
{
  return graph.get_vertices().size();
}

extern "C"
int* getEdgePairList()
{
  return (int*)&graph.edges[0];
}
#endif