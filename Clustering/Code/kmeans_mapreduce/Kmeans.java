package org.myorg;
 	
import java.io.IOException;
import java.util.*;
import java.text.DecimalFormat;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.BufferedWriter;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;

import org.apache.commons.collections.ListUtils;

public class Kmeans {

   static DecimalFormat decimal_format = new DecimalFormat("#.00");
   static long timestamp = System.currentTimeMillis();
   static List<String> initial_centroids = new ArrayList<>();;

   public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
     //private final static IntWritable one = new IntWritable(1);
     private Text word = new Text();

     public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
	   List<String[]> list_of_centroids = new ArrayList<>();
	   for(int i=0; i<initial_centroids.size(); i++){
	   		list_of_centroids.add(initial_centroids.get(i).split("\t"));
	   }
	   String input = value.toString();
	   String[] temp = input.split("\t");
	   String[] feature = new String[temp.length-2];
	   for(int i=2; i<temp.length; i++){
	   	feature[i-2] = temp[i];
	   }
	   double min_distance = -1;
	   double curr_distance = 0;
	   String[] curr_centroid = null;
	   String[] min_centroid = null;
	   for(int i=0; i<list_of_centroids.size(); i++){
	   	curr_centroid = list_of_centroids.get(i);
	   	curr_distance = 0;
	   	for(int j=0; j<curr_centroid.length; j++){
	   		curr_distance = curr_distance + Math.pow(Double.parseDouble(feature[j])-Double.parseDouble(curr_centroid[j]), 2);
	   	}
	   	curr_distance = Math.sqrt(curr_distance);
	   	if(min_distance == -1){
	   		min_distance = curr_distance;
	   		min_centroid = curr_centroid;
	   	}else{
	   		if(curr_distance < min_distance){
	   			min_distance = curr_distance;
	   			min_centroid = curr_centroid;
	   		}
	   	}
	   }
       output.collect(new Text(Arrays.toString(min_centroid)), new Text(Arrays.toString(feature)));
     }
   }

   public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
     public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
     	//System.out.println("Reducer reached for centroid "+key);
     	double[] acc_feature = null;
     	String[] temp = null;
     	String feature = null;
     	int no_of_items = 0;
     	while(values.hasNext()){
     		no_of_items++;
     		feature = values.next().toString();
     		feature = feature.replace("[","");
     		feature = feature.replace("]","");
     		temp = feature.split(",");
     		if(acc_feature == null){
     			acc_feature = new double[temp.length];
     			for(int i=0; i<temp.length; i++){
     				acc_feature[i] = Double.parseDouble(temp[i]);
     			}
     		}else{
     			for(int i=0; i<temp.length; i++){
     				acc_feature[i] = acc_feature[i]+Double.parseDouble(temp[i]);
     			}
     		}
     	}
     	for(int i=0; i<acc_feature.length; i++){
     		acc_feature[i] = Double.parseDouble(decimal_format.format(acc_feature[i]/no_of_items));
     	}
     	output.collect(new Text(Arrays.toString(acc_feature)), new Text("NEW_CENTROID"));
      }
   }

   public static void main(String[] args) throws Exception {

     Path pt = new Path(args[0]);
 	 FileSystem fs = FileSystem.get(new Configuration());
 	 BufferedReader br = null;
 	 String line = null;
 	 if(fs.exists(pt)){
 	 	 System.out.println("Input file read");
 	 	 br=new BufferedReader(new InputStreamReader(fs.open(pt)));
	     line=br.readLine();
	     int max_cluster_no = -1;
	     int cluster_no = -1;
	     String[] features = null;
	     String temp = null;
	     List<String> list_of_features = new ArrayList<>();
	     while (line != null){
	       features = line.split("\t");
	       temp = features[2];
	       for(int i=3; i<features.length; i++){
	       		temp = temp + "\t" + decimal_format.format(Double.parseDouble(features[i]));
	       }
	       list_of_features.add(temp);
	       cluster_no = new Integer(features[1]);
	       if(max_cluster_no == -1){
	       	max_cluster_no = cluster_no;
	       }
	       else{
	       	if(cluster_no > max_cluster_no){
	       		max_cluster_no = cluster_no;
	       	}
	       } 
	       line=br.readLine();
	     }
	     System.out.println("Total records in file "+list_of_features.size());
	     System.out.println("Max cluster no in input file is "+max_cluster_no);

	     if(args.length == 2){
	     	 // user didn't give centroids, select random centroids
		     Random random = new Random();		     
		     for(int i=0; i<max_cluster_no; i++){
		     	initial_centroids.add(list_of_features.get(random.nextInt(list_of_features.size())));
		     }
		 }else{
		 	String[] ids = args[2].split(",");
		 	for(int i=0; i<ids.length; i++){
		 		initial_centroids.add(list_of_features.get(Integer.parseInt(ids[i])-1));
		 	}
		 }
	     System.out.println("Initial centroids selected");
	     for(int i=0; i<initial_centroids.size(); i++){
	     	System.out.println(initial_centroids.get(i));
	     }
 	 }else{
 	 	System.out.println("File not found");
 	 	return;
 	 }

     int max_iterations = 100;
 	 int iteration_no = 1;
 	 int start_index = -1;
 	 int end_index = -1;
 	 List<String> new_centroids = new ArrayList<>();
 	 int no_of_matches = 0;
 	 String curr_centroid = null;
 	 String new_centroid = null;

 	 while(true){

 	 	if(iteration_no > max_iterations){
 	 		System.out.println("Centroids after Max Iteration");
        	for(int i=0; i<initial_centroids.size(); i++){
        		System.out.println(initial_centroids.get(i));
        	}
        	System.out.println("Output at directory "+args[1]+String.valueOf(timestamp)+(iteration_no-1));
 	 		break;
 	 	}

 	 	 JobConf conf = new JobConf(Kmeans.class);
	     conf.setJobName("kmeans"+iteration_no);

	     conf.setOutputKeyClass(Text.class);
	     conf.setOutputValueClass(Text.class);

	     conf.setMapperClass(Map.class);
	     // conf.setCombinerClass(Reduce.class);
	     conf.setReducerClass(Reduce.class);

	     conf.setInputFormat(TextInputFormat.class);
	     conf.setOutputFormat(TextOutputFormat.class);

	     FileInputFormat.setInputPaths(conf, new Path(args[0]));
	     FileOutputFormat.setOutputPath(conf, new Path(args[1]+String.valueOf(timestamp)+iteration_no));

	     JobClient.runJob(conf);
	     
	     System.out.println("Iteration "+iteration_no+" completed");
	     
	     pt = new Path(args[1]+String.valueOf(timestamp)+iteration_no+"/part-00000");
 	 	 fs = FileSystem.get(new Configuration());
 	 	 if(fs.exists(pt)){
 	 	 	br=new BufferedReader(new InputStreamReader(fs.open(pt)));
	     	line=br.readLine();
	     	while (line != null){
	     		start_index = line.indexOf("[");
	     		end_index = line.indexOf("]");
	     		line = line.substring(start_index+1, end_index);
	     		line = line.replaceAll(", ","\t");
	     		new_centroids.add(line);
	       		line=br.readLine();
	        }
	        for(int i=0; i<initial_centroids.size(); i++){
	        	curr_centroid = initial_centroids.get(i);
	        	for(int j=0; j<new_centroids.size(); j++){
	        		new_centroid = new_centroids.get(j);
	        		if(curr_centroid.equals(new_centroid)){
	        			no_of_matches++;
	        			break;
	        		}
	        	}
	        }
	        if(no_of_matches == initial_centroids.size()){
	        	System.out.println("Convergence reached");
	        	System.out.println("Centroids");
	        	for(int i=0; i<new_centroids.size(); i++){
	        		System.out.println(new_centroids.get(i));
	        	}
	        	System.out.println("Output at directory "+args[1]+String.valueOf(timestamp)+iteration_no);
	        	break;
	        }else{
	        	 initial_centroids.clear();
			     for(int i=0; i<new_centroids.size(); i++){
			     	initial_centroids.add(new_centroids.get(i));
			     }
			     System.out.println("Centroids updated");
			     iteration_no++;
		 	 	 new_centroids.clear();
		 	 	 no_of_matches = 0;
	        }

 	 	 }else{
 	 	 	System.out.println("Output file not found");
 	 	 	break;
 	 	 }
 	}
  }
} 	