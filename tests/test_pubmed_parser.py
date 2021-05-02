import unittest
from preprocessing.pubmed_parser import PubMedParser
import os


class PubMedParserTest(unittest.TestCase):
    def test_parseText(self):
        parser = PubMedParser()
        parser.parse_pubmed_from("./ressources/pubmed21n0001-small.xml.gz", "testfile.txt")
        # TODO: Mock Filesystem interaction
        with open('testfile.txt', 'r') as file:
            file_content = file.read()
            self.assertEqual(
                "(--)-alpha-Bisabolol has a primary antipeptic action depending on dosage, which is not caused by an alteration of the pH-value. The proteolytic activity of pepsin is reduced by 50 percent through addition of bisabolol in the ratio of 1/0.5. The antipeptic action of bisabolol only occurs in case of direct contact. In case of a previous contact with the substrate, the inhibiting effect is lost.\n"
                "A report is given on the recent discovery of outstanding immunological properties in BA 1 [N-(2-cyanoethylene)-urea] having a (low) molecular mass M = 111.104. Experiments in 214 DS carcinosarcoma bearing Wistar rats have shown that BA 1, at a dosage of only about 12 percent LD50 (150 mg kg) and negligible lethality (1.7 percent), results in a recovery rate of 40 percent without hyperglycemia and, in one test, of 80 percent with hyperglycemia. Under otherwise unchanged conditions the reference substance ifosfamide (IF) -- a further development of cyclophosphamide -- applied without hyperglycemia in its most efficient dosage of 47 percent LD50 (150 mg kg) brought about a recovery rate of 25 percent at a lethality of 18 percent. (Contrary to BA 1, 250-min hyperglycemia caused no further improvement of the recovery rate.) However this comparison is characterized by the fact that both substances exhibit two quite different (complementary) mechanisms of action. Leucocyte counts made after application of the said cancerostatics and dosages have shown a pronounced stimulation with BA 1 and with ifosfamide, the known suppression in the post-therapeutic interval usually found with standard cancerostatics. In combination with the cited plaque test for BA 1, blood pictures then allow conclusions on the immunity status. Since IF can be taken as one of the most efficient cancerostatics--there is no other chemotherapeutic known up to now that has a more significant effect on the DS carcinosarcoma in rats -- these findings are of special importance. Finally, the total amount of leucocytes and lymphocytes as well as their time behaviour was determined from the blood picture of tumour-free rats after i.v. application of BA 1. The thus obtained numerical values clearly show that further research work on the prophylactic use of this substance seems to be necessary and very promising.\n"
                "", file_content)
        os.remove("testfile.txt")
        self.assertEqual(0, parser._abstract_truncated_counter)
        self.assertEqual(0, parser._abstract_truncated_at_250_words_counter)
        self.assertEqual(0, parser._abstract_truncated_at_400_words_counter)


if __name__ == '__main__':
    unittest.main()
