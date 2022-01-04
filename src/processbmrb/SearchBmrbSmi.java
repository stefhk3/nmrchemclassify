package processbmrb;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.URL;
import java.util.Scanner;

import org.openscience.cdk.AtomContainer;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.io.MDLV2000Reader;
import org.openscience.cdk.smiles.SmiFlavor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

/**
 * Downloads bmrb smiles. Creates files in /tmp
 *
 */
public class SearchBmrbSmi {
	public static void main(String[] args) throws IOException, CDKException {
        URL url = new URL("https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/"); 
		Scanner sc = new Scanner(url.openStream());
		PrintWriter out = new PrintWriter("/tmp/test.smi");
	    SmilesGenerator sg = new SmilesGenerator(SmiFlavor.Generic);
		int i=0;
		while(sc.hasNextLine()) {
			String id=sc.nextLine();
			try {
				if(id.indexOf("bms")>-1) {
					id=id.substring(id.indexOf("bms"));
					id=id.substring(0,id.indexOf("/\">"));
					System.out.println(id);
					String urlmolstring="https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/"+id+"/"+id+".sdf";
					StringBuffer mol=new StringBuffer();
					URL urlmol = new URL(urlmolstring);
					Scanner sc2 = new Scanner(urlmol.openStream());
					while(sc2.hasNextLine()) {
						String line = sc2.nextLine();
						mol.append(line+"\r\n");
					}
					mol.append("> <BMRBID>\r\n"+id+"\r\n\r\n");
					mol.append("$$$$\r\n");
					sc2.close();
				    MDLV2000Reader mdlreader = new MDLV2000Reader(new StringReader(mol.toString()));
				    IAtomContainer cdkmol = (IAtomContainer) mdlreader.read(new AtomContainer());
				    AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(cdkmol);
				    Aromaticity.cdkLegacy().apply(cdkmol);
				    mdlreader.close();
					out.write(id+"\t"+sg.create(cdkmol)+"\r\n");
					System.out.println(id+";"+sg.create(cdkmol)+"\r\n");
					i++;
					//if(i>100)
					//	break;
				}
			}catch(Exception ex) {
				ex.printStackTrace();
			}
		}	
		sc.close();
		out.close();
	}

}
