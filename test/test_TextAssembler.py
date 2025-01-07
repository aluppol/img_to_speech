import unittest
import json

from src.TextAssembler import TextAssembler
from src.LabelTransformer import LabelTransformer
from src.TextClassifier import ClassifiedText


class TestTextAssembler(unittest.TestCase):
    def test_integration_1(self):
        with open('statics/model_training_data/roadto/10-ch1.json', 'r') as file:
            training_data = json.load(file)
        lt = LabelTransformer()
        classified_text = [ClassifiedText(lt.to_int(td['label']), td['text']) for td in training_data]
        text_assembler = TextAssembler()
        chapters_generator = text_assembler.process_classified_text(classified_text)
        for _ in chapters_generator:
            pass

        text_assembler.save_chapter()
        self.assertEqual(text_assembler.chapter.title, "1TH EABAN DON ED ROAD")
        self.assertEqual(
            text_assembler.chapter.text,
            'A programme whose basic thesis is, not that the system offree enterprise for profit has failed in this generation,'
             + ' but thatit has not yet been tried.\nF. D. Roosevelt.\n When the course of civilisation takes an unexpected turn,'
             +' wheninstead of the continuous progress which we have come toexpect, we find ourselves threatened by evils associated '
             + 'by uswith past ages of barbarism, we blame naturally anything butourselves.  Have we not all striven according to our '
             + 'best lights,and have not many of our finest minds incessantly worked tomake this a better world?  Have not all our'
             + ' efforts and hopes beendirected towards greater freedom, justice, and prosperity?  If theoutcome is so different from '
             + 'our aims, if, instead offreedom andprosperity, bondage and misery stare us in the face, is it not clearthat sinister '
             + 'forces must have foiled our intentions, that we arethe victims of some evil power which must be conquered beforewe can '
             + 'resume the road to better things?  However much we maydiffer when we name the culprit, whether it is the wicked')
        
    def test_integration_2(self):
        lt = LabelTransformer()
        text_assembler = TextAssembler()

        with open('statics/model_training_data/roadto/12-ch7.json', 'r') as file:
            training_data = json.load(file)
        classified_text = [ClassifiedText(lt.to_int(td['label']), td['text']) for td in training_data]
        chapters_generator = text_assembler.process_classified_text(classified_text)
        for _ in chapters_generator:
            pass
        
        with open('statics/model_training_data/roadto/13-ch7.json', 'r') as file:
            training_data = json.load(file)
        classified_text = [ClassifiedText(lt.to_int(td['label']), td['text']) for td in training_data]
        chapters_generator = text_assembler.process_classified_text(classified_text)
        for _ in chapters_generator:
            pass

        text_assembler.save_chapter()

        self.assertEqual(text_assembler.chapter.text, 'The control of the production of wealth is the control ofhuman life itself.\n'
                         + 'Hilaire Bel/oc.\n Most planners who have seriously considered the practicalaspects of their task have little '
                         + 'doubt that a directed economymust be run on more or less dictatorial lines.  That the complexsystem of interrelated'
                         + ' activities, if it is to be consciously directedat all, must be directed by a single staff of experts, and '
                         + 'thatultimate responsibility and power must rest in the hands of acommander-in-chief, whose actions must not '
                         + 'be fettered bydemocratic procedure, is too obvious a consequence of under-lying ideas of central planning not '
                         + 'to command fairly generalassent.  The consolation our planners offer us is that this authori-tarian direction '
                         + 'will apply "only" to economic matters.  One ofthe most prominent American planners, Mr.  Stuart Chase, assuresus, '
                         + 'for instance, that in a planned society "political democracycan remain if it confines itself to all but economic matter". '
                         + ' Suchassurances are usually accompanied by the suggestion that bygiving up freedom in what are, or ought to be, '
                         + 'the less import-ant aspects of our lives, we shall obtain greater freedom in thepursuit of higher values.  '
                         + 'On this ground people who abhor theidea of a political dictatorship often clamour for a dictator inthe economic '
                         + 'field. The arguments used appeal to our best instincts and oftenattract the finest minds.  If planning really did '
                         + 'free us from theless important cares and so made it easier to render our existenceone of plain living and high '
                         + 'thinking, who would wish tobelittle such an ideal?  If our economic activities really concernedonly the inferior '
                         + 'or even more sordid sides of life, of course weought to endeavour by all means to find a way to relieve our-selves '
                         + 'from the excessive care for material ends, and, leavingthem to be cared for by some piece of utilitarian machinery, '
                         + 'setour minds free for the higher things of life. Unfortunately the assurance people derive from this beliefthat the '
                         + 'power which is exercised over economic life is a powerover matters of secondary importance only, and which makesthem '
                         + 'take lightly the threat to the freedom of our economicpursuits, is altogether unwarranted.  It is largely a consequence '
                         + 'ofthe erroneous belief that there are purely economic ends separ-ate from the other ends of life.  Yet, apart from the '
                         + 'pathologicalcase of the miser, there is no such thing.  The ultimate ends of theactivities ofreasonable beings are '
                         + 'never economic.  Strictly speak-ing there is no "economic motive" but only economic factorsconditioning our striving '
                         + 'for other ends.  What in ordinarylanguageismisleadinglycalledthe"economicmotive"means merely the desire for general '
                         + 'opportunity, the desire forpower to achieve unspecified ends.  Cf. L. Robbins, The Economic Causes of War, 1939, '
                         + 'Appendix.  If we strive for money it isbecause it offers us the widest choice in enjoying the fruits of')
        
        