package processbmrb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.io.filefilter.WildcardFileFilter;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.io.iterator.IteratingSDFReader;

import com.google.common.io.Files;

/**
 * This reads a certain tag from the classyfire files, creates directories, and copies the images accordingly.
 * For each compound, only one spectrum is used.
 * Change the classyfire variable to run it for a different classyfire tag.
 *
 */
public class MakeFoldersSingle {
	public static void main(String[] args) throws IOException {
		String classyfire="Superclass";
		File dir=new File("./classyfire");
		File[] files=dir.listFiles();
		File outdirhmbc=new File("./classessingle/"+classyfire+"/hmbc");
		if(!outdirhmbc.exists())
			outdirhmbc.mkdir();
		File outdirhsqc=new File("./classessingle/"+classyfire+"/hsqc");
		if(!outdirhsqc.exists())
			outdirhsqc.mkdir();
		File outdirhmbctrain=new File("./classessingle/"+classyfire+"/hmbc/train");
		if(!outdirhmbctrain.exists())
			outdirhmbctrain.mkdir();
		File outdirhsqctrain=new File("./classessingle/"+classyfire+"/hsqc/train");
		if(!outdirhsqctrain.exists())
			outdirhsqctrain.mkdir();
		//read mapping files
		File inchiout=new File("./inchimap");
		BufferedReader br=new BufferedReader(new FileReader(inchiout));
		String line="";
		Map<String,String> ids=new HashMap<>(); 
		while((line=br.readLine())!=null){
			String[] id = line.split(";");
			ids.put(id[0],id[1]);
		}
		br.close();
		for(File file : files) {
			if(file.getName().endsWith(".sdf")) {
				IteratingSDFReader iter = new IteratingSDFReader(new FileInputStream(file), DefaultChemObjectBuilder.getInstance());
			    int molcount = 0;
			    while (iter.hasNext()) {
			      IAtomContainer mol = iter.next();
			      //get the property
			      String term = mol.getProperty(classyfire);
			      term=term.replace("/", "-");
			      String inchikey = mol.getProperty("InChIKey"); 
			      inchikey=inchikey.split("=")[1];
			      System.out.println(inchikey);
			      //get bmse id
			      String bmseid=ids.get(inchikey);
			      //create output image dir if not exists
			      File imagedirhsqc=new File("./classessingle/"+classyfire+"/hsqc/train/"+term);
			      if(!imagedirhsqc.exists())
			    	  imagedirhsqc.mkdirs();
			      File imagedirhmbc=new File("./classessingle/"+classyfire+"/hmbc/train/"+term);
			      if(!imagedirhmbc.exists())
			    	  imagedirhmbc.mkdirs();
			      //copy from images hmbc
			      FileFilter fileFilter = new WildcardFileFilter("(2D-HMBC)_"+bmseid+"_nmr_*.png");
			      File imagesDir=new File("./images/");
			      File[] filesimagehmbc = imagesDir.listFiles(fileFilter);
			      if(filesimagehmbc.length>0) {
			    	  System.out.println(filesimagehmbc[0]);
			    	  File fileto=new File(imagedirhsqc.getPath()+"/"+filesimagehmbc[0].getName());
			    	  if(!fileto.exists())
			    		  Files.copy(filesimagehmbc[0], fileto);
			      }
			      //copy from images hsqc
			      fileFilter = new WildcardFileFilter("(2D-HSQC)_"+bmseid+"_nmr_*.png");
			      File[] filesimagehsqc = imagesDir.listFiles(fileFilter);
			      if(filesimagehsqc.length>0) {
			    	  System.out.println(filesimagehsqc[0]);
			    	  File fileto=new File(imagedirhmbc.getPath()+"/"+filesimagehsqc[0].getName());
			    	  if(!fileto.exists())
			    		  Files.copy(filesimagehsqc[0], fileto);
			      }
			      molcount++;
			      //if(molcount==10)
			      //	  break;
			    }
			    iter.close();
			}
		}
		File outdirhmbctest=new File("./classessingle/"+classyfire+"/hmbc/test");
		if(!outdirhmbctest.exists())
			outdirhmbctest.mkdir();
		File outdirhsqctest=new File("./classessingle/"+classyfire+"/hsqc/test");
		if(!outdirhsqctest.exists())
			outdirhsqctest.mkdir();
		File[] terms=outdirhmbctrain.listFiles();
		for(File termfile : terms) {
			File[] images=termfile.listFiles();
			File outdirhmbctestterm=new File("./classessingle/"+classyfire+"/hmbc/test/"+termfile.getName());
			if(!outdirhmbctestterm.exists())
				outdirhmbctestterm.mkdir();
			int testlen=(int)(images.length*.2);
			for(int i=0;i<testlen;i++) {
				Files.move(images[i],new File(outdirhmbctestterm.getPath()+"/"+images[i].getName()));
			}
		}
		terms=outdirhsqctrain.listFiles();
		for(File termfile : terms) {
			File[] images=termfile.listFiles();
			File outdirhsqctestterm=new File("./classessingle/"+classyfire+"/hsqc/test/"+termfile.getName());
			if(!outdirhsqctestterm.exists())
				outdirhsqctestterm.mkdir();
			int testlen=(int)(images.length*.2);
			for(int i=0;i<testlen;i++) {
				Files.move(images[i],new File(outdirhsqctestterm.getPath()+"/"+images[i].getName()));
			}
		}
	}
}
